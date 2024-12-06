from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm import tqdm
import re
import asyncio
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown, nvmlDeviceGetMemoryInfo
import time
from kiwipiepy import Kiwi


# TODO STT.__batchnum에서 gpu 메모리에 따라 batch 갯수 반환하게 코드 작성 필요 (현재는 고정값 4)

'''
requirements

pip install faster-whisper==1.1.0 
pip install nvidia-ml-py3
pip install kiwipiepy


'''


'''
사용법 (@ main)

STT() 선언
STT.modelinit()으로 모델변수 선언
    model_path : 모델 파일이 담긴 디렉토리
        model_path
            ㄴ config.json
            ㄴ model.bin
            ㄴ preprocessor_config.json
            ㄴ tokenizer.json
            ㄴ vocabulary.json

    device : "cpu" / "cuda"
    compute_type : float_16 
STT.transcribe("오디오 파일 위치", log_gpu, log_file) 실행 => 문장이 담긴 list 반환
    log_gpu : gpu 사용량 로그 남길지 여부
    log_file : 로그 파일 위치, log_gpu = False 인 경우 기입 안해도 됨



e,g)

# logging 기능은 nvidia gpu에서만 작동됨
# STT.transcribe는 문장들의 list 반환함

from STT import STT

STTpipeline = STT()
STTpipeline.modelinit(model_path, "cuda", "float_16")
STTresult = STTpipeline.transcribe(audio_path, True, log_file)



'''

class STT:

    __model = None
    __device = "cpu"


    def __batchnum(self) -> int:
        if self.__model == "cpu":
            return 1
        else:
        #    pass
            return 4

    def __parsesent(self, input : list[str]) -> list[str]:
        kiwi = Kiwi()
        result = [sent.text for sent in kiwi.split_into_sents(" ".join(input))]
        return result
    

    def modelinit(self, model_path, device, compute_type) -> None:
        self.__device = device
        self.__model = WhisperModel(model_path, device = device, compute_type = compute_type)

    async def loggpuuse(self, log_file, interval) -> bool:
        nvmlInit()
        gpu_handle = nvmlDeviceGetHandleByIndex(0)
        maxuse = 0

        with open(log_file, "w") as log:
            log.write("Timestamp, GPU Utilization (%), Memory Utilization (%)\n")
            loginfo = []
            starttime = 0
            try:
                while True:  # Regularly check the stop event
                    utilization = nvmlDeviceGetUtilizationRates(gpu_handle)
                    gpu_util = utilization.gpu  # GPU utilization in %
                    mem_util = utilization.memory  # Memory utilization in %
                    memory_info = nvmlDeviceGetMemoryInfo(gpu_handle)
                    used_memory_mb = int(memory_info.used / (1024 ** 2))
                    if used_memory_mb > maxuse:
                        maxuse = used_memory_mb
                    
                    # Log data with timestamp
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    loginfo.append(f"time: {timestamp}, gpu utilization: {gpu_util}%, Memory: {used_memory_mb}MB ({mem_util}%)\n")
                    print("added log")

                    await asyncio.sleep(interval)  # Wait before checking again
            except asyncio.CancelledError:
                pass  # Allow for graceful cancellation
            finally:
                log.write(f"Elapsed : {time.time()-starttime}, Max usage: {maxuse}MB\n")
                for log_entry in loginfo:
                    log.write(log_entry)
                nvmlShutdown()
                return True  # Clean up NVML


    async def __transcribeaudiofile(self, audio_path) -> list[str]:
        if self.__model is None:
            raise ValueError("Model Not Initialized")
        batch_num = self.__batchnum()

        if self.__device == "cpu":
            result,_ = await asyncio.to_thread(self.__model.transcribe, audio_path, language="ko")
        else:
            batched_model = BatchedInferencePipeline(model=self.__model)
            result,_ = await asyncio.to_thread(batched_model.transcribe, audio_path, batch_size=batch_num)
        
        transcription = []
        for segment in tqdm(await asyncio.to_thread(list,result)):
            transcription.append(segment.text)
        
        return self.__parsesent(transcription)
        #return transcription
    
    
    def transcribe(self, audio_path: str, log_gpu: bool = False, log_file : str = None) -> list[str]:

        if (log_gpu == True) and (log_file is None):
            raise ValueError("No log file given")

        if self.__device == 'cpu':
            segments, info = self.__model.transcribe(audio_path, beam_size=5)

            return list(segments)

        loop = asyncio.get_event_loop()

        # Create a stop event to signal the logger task to stop

        # Start the logger task if log_gpu is True
        logger = loop.create_task(self.loggpuuse(log_file, 5)) if log_gpu else None
        # Start the transcriber task
        transcriber = loop.create_task(self.__transcribeaudiofile(audio_path))
        #transcriber = loop.create_task(self.wait())

        result = loop.run_until_complete(transcriber)
        
        logger.cancel()
        try:
            loop.run_until_complete(logger)
        except asyncio.CancelledError:
            pass  # A was cancelled after 10 seconds, we handle that

        if loop.is_closed:
            return result  # Return the result of the transcriber task

def transcribe_audio(audio_path, model_path, device):
    STTpipeline = STT()
    if device == 'cuda':
        STTpipeline.modelinit(model_path, device, 'float16') # 기본적으로 float 16 연산 활용
    else:
        STTpipeline.modelinit(model_path, device, 'int8')

    output = STTpipeline.transcribe(audio_path, False)
    return output


def save_segments_to_txt(segments, file_name):
    """
    주어진 데이터 형식에 맞춰 파일로 저장하는 함수.
    
    Parameters:
        segments (list): 데이터 리스트 (딕셔너리 형식으로 구성)
        file_name (str): 저장할 파일 이름
    """
    try:
        with open(file_name, "w", encoding="utf-8") as file:
            for segment in segments:
                file.write(f"{{'String': '{segment.text}', 'Start': {segment.start}, 'End': {segment.end}}}\n")
        print(f"Segments가 '{file_name}' 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")