# AI-Detection Optimization
Variants: 
* https://huggingface.co/umm-maybe/AI-image-detector for non SDXL
* https://huggingface.co/Organika/sdxl-detector for SDXL

Findings:
* It may not always be distinguishable by humans what may be labeled as "AI detected" or not
* The model tested mostly (sdxl-detector) still has potential for improvement. Although it performed well on initially generated images.
* It is possible to optimize against detection for both a specific restricted embedding space and random embeddings. Both reached above 90% and sometimes 100% human-likedness.

## Trying to create random images not detected as AI

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/5e4f948c-9924-40d8-8ede-fab570cbe8ce

Parameters
```python
population_size = 100
num_generations = 100
batch_size = 1
elitism = 1
inference_steps = 1

embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()
creator = SDXLPromptEmbeddingImageCreator(batch_size=batch_size, inference_steps=inference_steps)
evaluator = AIDetectionImageEvaluator()
crossover = PooledArithmeticCrossover(crossover_rate=0.5, crossover_rate_pooled=0.5)
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.05, mutation_strength=3, 
                                                     clamp_range=(embedding_range.minimum, embedding_range.maximum)) 
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.05, mutation_strength=0.7, 
                                                            clamp_range=(pooled_embedding_range.minimum, pooled_embedding_range.maximum))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)

# Prepare initial arguments, random population of *reasonable* prompt embeddings
init_args = [PooledPromptEmbedData(embedding_range.random_tensor_in_range(), pooled_embedding_range.random_tensor_in_range()) 
             for _ in range(population_size)]

# Create and run the genetic algorithm
ga = GeneticAlgorithm(
    population_size=population_size,
    num_generations=num_generations,
    solution_creator=creator,
    evaluator=evaluator,
    mutator=mutator,
    crossover=crossover,
    selector=selector,
    initial_arguments=init_args,
    elitism_count=elitism,
    post_evaluation_callback=save_images_post_evaluation,
)
```

Sadly the notebook file went missing for this experiment.

## Optimizing specific Prompt Embedding Space to avoid AI detection
Starting the optimization with prompt embeddings weighted 95% the direction of the prompt and 5% random embeddings. The first generation was mostly detected (>90%) as AI. After around 100 generations the first image with 99%+ fitness (human-likedness) seen below was generated. The prompt was "cartoon monster howling at the moon in a spooky forest".

This experiment was also tried with CLIPScore + AI-Detection, but lead to weird style transitions because of the mismatch between style direction of the generational (SDXL-Turbo) and CLIP (vit-base-patch16) model. Restricting the embedding space from the initial population on seemed to work better in mainaining style.

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/481531f3-b125-472b-9c32-385f24a5c81e

![fitness_aid_prompt](https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/9991fef8-aa92-4760-b6c8-6d2617a90fad)

| Starting Generation | Best Image 99%+ Human-Likedness |
| ------------------- | ----------------------- |
|         <img width="300" alt="sdxl_monster_start" src="https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/747208e3-9af0-4b02-9d27-143924e72f98">            |             <img width="300" alt="sdxl_monster_final" src="https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/dc3fbb65-62f6-4994-b045-2270f4bf7807">            |

```python
population_size = 100
num_generations = 200
batch_size = 1
elitism = 1
inference_steps = 4
crossover_proportion = 0.7
crossover_rate = 0.9
mutation_rate = 0.2
strict_osga = False
prompt = "cartoon monster howling at the moon in a spooky forest"

# Define min/max values for the prompt embeddings
embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()
# Try low mutation, high weight for starting prompt, to not deviate too much from the prompt
creator = SDXLPromptEmbeddingImageCreator(batch_size=batch_size, inference_steps=inference_steps)
evaluator = AIDetectionImageEvaluator()
crossover = PooledArithmeticCrossover(interpolation_weight=0.5, interpolation_weight_pooled=0.5, 
                                      proportion=crossover_proportion, proportion_pooled=crossover_proportion)
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.05, mutation_strength=1.5, 
                                                     clamp_range=(embedding_range.minimum, embedding_range.maximum)) 
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.05, mutation_strength=0.2, 
                                                            clamp_range=(pooled_embedding_range.minimum, pooled_embedding_range.maximum))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)

# Prepare initial arguments, random population of *reasonable* prompt embeddings mixed with the initial prompt
init_crossover = PooledArithmeticCrossover(interpolation_weight=0.95, interpolation_weight_pooled=0.95)
initial_prompt_arg = creator.arguments_from_prompt(prompt) 
init_args = [init_crossover.crossover(initial_prompt_arg,  
                                      PooledPromptEmbedData(embedding_range.random_tensor_in_range(), pooled_embedding_range.random_tensor_in_range())) 
                 for _ in range(population_size)]
```

[View the full notebook](./ga_aidetection_monster)

