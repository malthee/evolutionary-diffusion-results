# CLIP-Score Optimization
CLIP-Score https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html

## Trying to "Find" a Prompt from Random Embeddings
Initializing the population with random embeddings and measuring fitness with CLIP-Score containing the prompt.  
**Prompt:** "surreal simulation of the universe, fantasy crazy unimaginable 4k high quality ultra hd beautiful"

TODO video

![CLIPPromptOptimization](./ga_100gen_200pop_surreal.png)

Parameters
```python
population_size = 100
num_generations = 200
batch_size = 1
elitism = 1
inference_steps = 3
prompt = "surreal simulation of the universe, fantasy crazy unimaginable 4k high quality ultra hd beautiful"

embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()
creator = SDXLPromptEmbeddingImageCreator(batch_size=batch_size, inference_steps=inference_steps)
evaluator = CLIPScoreEvaluator(prompt=prompt) 
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

| Optimized Image | Images directly generated with prompt |
| --- | --- |
| ![OptimizedImageSurreal](./surreal_result.png) | ![PromptImageSurreal](./surreal_comparison.png) |

The optimized image is a bit more colorful and fits the CLIP-Score a few points better than the prompt images. Quality-wise the prompt images seem to be a bit better.

[View the full notebook](./ga_100gen_200pop_surreal.ipynb)

## Trying to Optimize for "Aesthetics" by using the CLIP-Score
The idea is to use the CLIP-Score as a fitness function to optimize for aesthetics by defining the prompt as "aesthetic".

TODO video

Parameters
```python
population_size = 100
num_generations = 200
batch_size = 1
elitism = 1
inference_steps = 3
prompt = "aesthetic"

embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()
creator = SDXLPromptEmbeddingImageCreator(batch_size=batch_size, inference_steps=inference_steps)
evaluator = CLIPScoreEvaluator(prompt=prompt) 
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

| Optimized Image | Images directly generated with prompt |
| --- | --- |
| ![OptimizedImageAesthetic](./ga_100gen_200pop_aesthetics.png) | ![PromptImageAesthetic](./aesthetics_comparison.png) |

Interesting is that both results are using a pinkish color palette. The optimized image is a bit "over the top" but fits the CLIP-Score a few points better than the prompt images.

[View the full notebook](./ga_100gen_200pop_aesthetics.ipynb)
