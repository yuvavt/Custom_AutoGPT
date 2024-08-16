import transformers
from huggingface_hub import HfApi, HfFolder
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CustomThinkingProcess:
    def __init__(self):
        # Initialize Hugging Face API client with API key from .env file
        hf_api_key = os.getenv('HF_API_KEY')  # Use 'HF_API_KEY' for consistency
        if not hf_api_key:
            raise ValueError("HF_API_KEY not found in environment variables")
        HfFolder.save_token(hf_api_key)
        
        self.tool_descriptions = {
            "Tool A": "A comprehensive materials science API that provides access to a wide range of materials data and properties. It can handle queries related to structural, mechanical, electrical, and thermal properties of various materials.",
            "Tool B": "PyThermoPhY, a Python library for thermodynamic calculations. It specializes in computing thermodynamic properties of substances and mixtures, phase equilibria, and reaction equilibria."
        }
        self.analysis_prompt = "Analyze the user's question in detail. Consider the following aspects:\n1. What is the main subject or domain of the question?\n2. What type of information or calculation is being requested?\n3. Are there any specific requirements or constraints mentioned?"
        self.comparison_prompt = "Compare your analysis of the user's question with the capabilities of Tool A and Tool B. Consider how well each tool's features align with the requirements of the question."
        self.decision_prompt = "Based on your analysis and comparison, decide which tool (A or B) is most appropriate for addressing the user's question. Explain your reasoning for this choice."
        self.action_planning_prompt = "Now that you've chosen a tool, outline a plan of action for using this tool to address the user's question. Describe the general steps you would take, without going into specific implementation details."

        # Define model
        model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

        # Load model configuration
        config = transformers.AutoConfig.from_pretrained(model_name)
        
        # Check and fix rope_scaling configuration if needed
        if 'rope_scaling' in config.to_dict():
            rope_scaling = config.rope_scaling
            # Ensure it has 'type' and 'factor' fields
            if 'type' not in rope_scaling or 'factor' not in rope_scaling:
                # Adjust the configuration as needed
                rope_scaling = {
                    'type': 'default_type',  # Provide appropriate value
                    'factor': 8.0            # Provide appropriate value
                }
                config.rope_scaling = rope_scaling

        # Initialize the pipeline with the adjusted configuration
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            config=config
        )
        print("Model created")

    def process(self, user_question):
        analysis = self.analyze(user_question)
        comparison = self.compare(analysis)
        decision = self.decide(comparison)
        action_plan = self.plan_action(decision)
        return action_plan

    def analyze(self, question):
        message = f"{self.analysis_prompt}\n\n{question}"
        response = self.pipeline(message, max_length=150)
        analysis = response[0]['generated_text']
        return analysis

    def compare(self, analysis):
        tools_description = "\n".join([f"{key}: {value}" for key, value in self.tool_descriptions.items()])
        message = f"{self.comparison_prompt}\n\nAnalysis: {analysis}\n\nTools: {tools_description}"
        response = self.pipeline(message, max_length=150)
        comparison = response[0]['generated_text']
        return comparison

    def decide(self, comparison):
        message = f"{self.decision_prompt}\n\nComparison: {comparison}"
        response = self.pipeline(message, max_length=150)
        decision = response[0]['generated_text']
        return decision

    def plan_action(self, decision):
        message = f"{self.action_planning_prompt}\n\nDecision: {decision}"
        response = self.pipeline(message, max_length=150)
        action_plan = response[0]['generated_text']
        return action_plan
