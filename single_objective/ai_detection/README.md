# AI-Detection Optimization
Variants: 
* https://huggingface.co/umm-maybe/AI-image-detector for non SDXL
* https://huggingface.co/Organika/sdxl-detector for SDXL


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


## TODO Optimizing specific Prompt to avoid AI detection
already possible for some instances, do not use NSGA-II with ClipScore, maybe try with smaller mutiation and start with prompt and a little variation
