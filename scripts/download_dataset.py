import os
from datasets import load_dataset

def download_and_save_dataset():
    """
    Télécharge le dataset 'bitext/Bitext-customer-support-llm-chatbot-training-dataset'
    et le sauvegarde dans data/
    """
    print("Chargement du dataset depuis HuggingFace...")
    ds = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset')
    
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11_1.csv")
    
    print(f"Sauvegarde du dataset train ({len(ds['train'])} exemples) vers {output_path}...")
    ds['train'].to_csv(output_path, index=False)
    
    print("Téléchargement et sauvegarde terminés avec succès !")

if __name__ == "__main__":
    download_and_save_dataset()
