We selected mT0-small (300M parameters) as our primary model and mT0-base (580M parameters) as our secondary model.

Both models belong to the BigScience mT0 family and support multilingual instruction following. Their relatively small sizes make them practical candidates for our computational resources.

The primary model will be used for zero-shot prompting, five-shot prompting, and LoRA fine-tuning. The secondary model will only be used for prompting experiments.

Using models of different sizes from the same family will allow us to investigate whether model size affects emotion classification performance in English and Spanish.
