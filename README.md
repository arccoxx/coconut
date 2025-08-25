Long Shot: Extending COCONUT: A Dynamic Graph-Based Approach to Continuous Latent Reasoning
This project is an advanced implementation and extension of the ideas presented in the Facebook AI Research paper, "COCONUT: Continuous Causal Transformer." While the original COCONUT introduced the concept of a "navigator" that operates in a continuous latent space to generate reasoning steps, this implementation evolves that concept into a more dynamic and robust cognitive architecture.

The core of this project is a CognitiveGraphModel that integrates a large language model (LLM) with a dynamic working memory managed by a graph attention network. This allows the model to build an explicit, graph-based representation of its reasoning process, where each "thought" is a node in the graph.

Key Architectural Features
This implementation introduces several key advancements over a basic continuous reasoning model:

Dynamic Graph Memory (GraphMemory): Instead of a simple sequential memory, this model uses a graph structure that is built on the fly for each problem. This allows for more complex, non-linear relationships between thoughts.

Graph Attention Navigator (GraphAttentionNavigator): The navigator is no longer a simple feed-forward network. It is a Graph Attention Network (GAT) that is deeply integrated with the memory. At each step, it uses an attention mechanism to read from all existing nodes in the graph, creating a context-aware "hyperedge" that informs the generation of the next thought.

Scheduled Sampling: To combat the "exposure bias" inherent in teacher-forcing, this script uses scheduled sampling. During training, the navigator is gradually weaned off the perfect ground-truth thoughts and forced to use its own (potentially imperfect) predictions, making it more robust during inference.

PEFT / LoRA for Stability: To prevent the "catastrophic forgetting" that can occur when fine-tuning large models, this implementation uses PEFT (specifically LoRA). This freezes the majority of the base model's weights and only fine-tunes a small set of adapter layers, preserving the model's powerful pre-trained knowledge.

High-Quality Teacher Signal: The model is trained using a structured teacher signal that generates logical reasoning steps based on the numbers and goals of the problem, providing a more effective guide for the navigator.

How It Works
Training
Initialization: For each problem, a GraphMemory is initialized with a starting state derived from the problem's prompt.

Reasoning Loop (Scheduled Sampling): The GraphAttentionNavigator iteratively generates a chain of "thought" vectors. At each step, it either uses the ground-truth thought from the teacher or its own previous output as the input for the next step, based on a decaying probability (epsilon).

Loss Calculation: The training is guided by a balanced, two-part loss function:

Navigator Loss (thought_loss): A cosine similarity loss that encourages the navigator's predicted thoughts to align with the direction of the teacher's thoughts.

LM Loss (lm_loss): A standard cross-entropy loss that trains the base language model to generate the correct final answer, conditioned on the (potentially imperfect) reasoning chain produced during the scheduled sampling loop.

Inference
Initialization: A GraphMemory is initialized from the user's prompt.

Autoregressive Reasoning: The GraphAttentionNavigator generates a full reasoning chain on its own, with each new thought being added to the memory and used as the input for the next step.

Final Answer Generation: The base language model receives the full prompt and the generated reasoning chain as its context and produces the final text answer.

Getting Started
1. Installation
This project relies on several key libraries, including unsloth for high-performance training.

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" "trl<0.9.0" "peft<0.11.0" "accelerate<0.30.0" "bitsandbytes<0.44.0"
pip install datasets

2. Hugging Face Authentication
To access the Llama 3 model, you'll need to authenticate with your Hugging Face account. Run the following in your Python environment and enter your access token when prompted.

from huggingface_hub import login
login()

3. Running the Training
The script is designed to be run directly. It will automatically download the necessary models and datasets, run a comprehensive suite of unit tests, and then begin the training process.

python current_model_teacher_forcing.py

You can configure the training parameters, such as the number of epochs and batch size, in the train() function at the bottom of the script.
