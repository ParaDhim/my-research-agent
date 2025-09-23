# fine_tune_model.py
import os
import torch
import streamlit as st
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline
)

# Configuration
MODEL_CHECKPOINT = "t5-small"
MODEL_SAVE_PATH = "t5-small-scientific-paper-summarizer"
DATASET_NAME = "franz96521/scientific_papers"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128

class ScientificPaperSummarizer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.summarizer = None
        
    @st.cache_resource
    def load_fine_tuned_model(_self):
        """Load the fine-tuned model if it exists, otherwise return None."""
        try:
            if os.path.exists(MODEL_SAVE_PATH):
                st.info("📖 Loading fine-tuned T5 summarizer model...")
                _self.summarizer = pipeline(
                    "summarization", 
                    model=MODEL_SAVE_PATH, 
                    tokenizer=MODEL_SAVE_PATH,
                    device=0 if torch.cuda.is_available() else -1
                )
                st.success("✅ Fine-tuned model loaded successfully!")
                return True
            else:
                st.warning("⚠️ Fine-tuned model not found. Use 'Train Model' to create one.")
                return False
        except Exception as e:
            st.error(f"❌ Error loading fine-tuned model: {str(e)}")
            return False
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize text using the fine-tuned model."""
        if not self.summarizer:
            return "Fine-tuned model not available. Please train the model first."
        
        try:
            # Truncate input text if too long
            if len(text) > MAX_INPUT_LENGTH * 4:  # Rough character estimate
                text = text[:MAX_INPUT_LENGTH * 4]
            
            result = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=30,
                do_sample=False
            )
            return result[0]['summary_text']
        except Exception as e:
            return f"Error generating summary: {str(e)}"

def preprocess_function(examples, tokenizer):
    """Tokenizes the text and abstracts."""
    inputs = tokenizer(
        examples["full_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["abstract"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length"
        )
    
    inputs["labels"] = labels["input_ids"]
    return inputs

def fine_tune_model():
    """Fine-tune the T5 model for scientific paper summarization."""
    
    # Create a progress container
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0, "🚀 Starting model fine-tuning...")
        status_text = st.empty()
        
        try:
            # Step 1: Load dataset
            status_text.text("📚 Loading scientific papers dataset...")
            progress_bar.progress(10)
            
            DATA_FILE_URL = "hf://datasets/franz96521/scientific_papers/scientific_paper_en.csv"
            dataset = load_dataset(
                "csv", 
                data_files={'train': DATA_FILE_URL}, 
                split='train[:1000]',
                cache_dir="./hf_cache"
            )
            dataset = dataset.train_test_split(test_size=0.1)
            
            # Step 2: Load tokenizer
            status_text.text("🔤 Loading tokenizer...")
            progress_bar.progress(20)
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
            
            # Step 3: Preprocess dataset
            status_text.text("🔄 Preprocessing and tokenizing dataset...")
            progress_bar.progress(30)
            
            tokenized_datasets = dataset.map(
                lambda examples: preprocess_function(examples, tokenizer), 
                batched=True
            )
            tokenized_datasets = tokenized_datasets.remove_columns(['id', 'full_text', 'abstract'])
            
            # Step 4: Load model
            status_text.text("🧠 Loading base T5 model...")
            progress_bar.progress(40)
            
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
            
            # Step 5: Setup training
            status_text.text("⚙️ Setting up training configuration...")
            progress_bar.progress(50)
            
            training_args = Seq2SeqTrainingArguments(
                output_dir=MODEL_SAVE_PATH,
                eval_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=2,  # Reduced for memory
                per_device_eval_batch_size=2,
                weight_decay=0.01,
                save_total_limit=3,
                num_train_epochs=2,  # Reduced for demo
                predict_with_generate=True,
                fp16=torch.cuda.is_available(),
                push_to_hub=False,
                report_to="none",
                logging_steps=50,
                save_steps=500,
            )
            
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # Step 6: Train model
            status_text.text("🎯 Training model... This may take several minutes...")
            progress_bar.progress(60)
            
            trainer.train()
            
            # Step 7: Save model
            status_text.text("💾 Saving fine-tuned model...")
            progress_bar.progress(90)
            
            trainer.save_model()
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            
            # Step 8: Test model
            status_text.text("🧪 Testing fine-tuned model...")
            progress_bar.progress(95)
            
            # Quick test
            summarizer = pipeline("summarization", model=MODEL_SAVE_PATH, tokenizer=MODEL_SAVE_PATH)
            sample_text = dataset['test'][0]['full_text'][:2000]
            test_summary = summarizer(sample_text)[0]['summary_text']
            
            progress_bar.progress(100)
            status_text.text("✅ Model fine-tuning completed successfully!")
            
            # Display results
            st.success("🎉 Fine-tuning completed!")
            
            with st.expander("📋 View Test Results", expanded=True):
                st.subheader("Sample Text (truncated):")
                st.text_area("Input", sample_text, height=150, disabled=True)
                
                st.subheader("Generated Summary:")
                st.text_area("Output", test_summary, height=100, disabled=True)
                
                st.subheader("Original Abstract:")
                st.text_area("Reference", dataset['test'][0]['abstract'], height=100, disabled=True)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error during fine-tuning: {str(e)}")
            progress_bar.progress(0)
            status_text.text("❌ Fine-tuning failed!")
            return False

def show_model_training_interface():
    """Show the model training interface in the sidebar."""
    st.markdown('<div class="section-header">🤖 AI Model Training</div>', unsafe_allow_html=True)
    
    # Check if model exists
    model_exists = os.path.exists(MODEL_SAVE_PATH)
    
    if model_exists:
        st.markdown('<div class="status-success">✅ Fine-tuned Model Available</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Retrain Model", type="secondary", use_container_width=True):
            if st.button("⚠️ Confirm Retrain", type="primary", use_container_width=True):
                with st.spinner("Retraining model..."):
                    fine_tune_model()
                st.rerun()
    else:
        st.markdown('<div class="status-warning">⚠️ No Fine-tuned Model</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model... This may take 10-30 minutes..."):
                if fine_tune_model():
                    st.rerun()

# Global instance
summarizer_instance = ScientificPaperSummarizer()