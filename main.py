from src.embed import Embedder
from src.utils import Utils
from src.annot import PDFAnnotator
from src.stt import transcribe_audio, save_segments_to_txt

import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os


def main():
    parser = argparse.ArgumentParser(description="Parse arguments for PDF to Text task")
    parser.add_argument('--pdf_path', type=str, required=True, help="Path to the PDF file")
    parser.add_argument('--audio_path', type=str, required=True, help="Path to the Audio file")
    parser.add_argument('--model_path', type=str, default='large-v3', help="Directory to the Model") # ../model에 가중치 추가해야 함
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    # parser.add_argument('--text_path', type=str, required=True, help="Path for the STT result file")
    parser.add_argument('--show', type=bool, default=False, help="Flag to show the embedded images (default: False)")
    parser.add_argument('--query', type=str, default='long', help="length of image description query (default: long)")
    args = parser.parse_args()

    # Speech To Text
    print('Transcribing audio to text...')
    script = transcribe_audio(args.audio_path, args.model_path, args.device) # script는 문장들의 리스트 
    save_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    save_segments_to_txt(script, os.path.join('logs', save_name + '.txt'))

    os.mkdir(f"./logs/{os.path.basename(args.pdf_path).split('.')[0]}_{datetime.now().strftime('%y%m%d%H%M%S')}")
    log_dir = f"./logs/{os.path.basename(args.pdf_path).split('.')[0]}_{datetime.now().strftime('%y%m%d%H%M%S')}"
    output_filename = os.path.join('results', f"{os.path.basename(args.pdf_path).split('.')[0]}_annotated_{datetime.now().strftime('%y%m%d%H%M%S')}.pdf")

    # Page Embedding
    print('Creating page indexes...')
    page_embeddings = Embedder.create_page_embeddings(args.pdf_path, show = args.show, query=args.query)

    # Sentence Embedding
    print('Creating sentence indexes...')
    sent_embeddings, strings_list, sig_idx = Embedder.create_sent_embeddings(os.path.join('logs', save_name + '.txt'))

    # Similarity Matrix Creation
    aligned = Utils.embedNalign(sig_idx, page_embeddings, sent_embeddings, length_cutoff=10)
    result_json = Utils.get_result(strings_list, aligned)
    with open(os.path.join(log_dir, f'alignment_{args.query}.json'), 'w', encoding='utf-8') as f:
        f.write(result_json)
        print(f'json file sucessfully saved to {os.path.join(log_dir, f"alignment_{args.query}.json")}')

    plt.scatter(sig_idx, [a[1] for a in aligned], c='blue')
    plt.savefig(os.path.join(log_dir, f'alignment_results.png'))

    #annot
    PDFAnnotator.add_summary(args.pdf_path, os.path.join(log_dir, f"alignment_{args.query}.json"), output_filename, log_dir, t = 'path')

if __name__ == '__main__':
    main()