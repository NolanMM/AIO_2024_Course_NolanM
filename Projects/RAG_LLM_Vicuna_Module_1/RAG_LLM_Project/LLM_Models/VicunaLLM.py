import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline


class vicuna_llm_model_implement:
    """
    ModelSetup Class

    The ModelSetup class is responsible for setting up the model, tokenizer,
    and text generation pipeline using the Vicuna model.

    Attributes:
    -----------
    model_pipeline : pipeline
        The text generation pipeline.
    llm : HuggingFacePipeline
        The language model pipeline for text generation.
    """

    def __init__(self):
        """
        Initializes the ModelSetup class and sets up the model and pipeline.
        """
        self.model_pipeline = None
        self.llm = None
        self.setup_model()

    def setup_model(self):
        """
        Sets up the model, tokenizer, and text generation pipeline.
        """
        # Model and tokenizer configuration
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        MODEL_NAME = "lmsys/vicuna-7b-v1.5"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=nf4_config,
            low_cpu_mem_usage=True
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

        self.model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            device_map="auto"
        )

        self.llm = HuggingFacePipeline(pipeline=self.model_pipeline)
