# Multi-Objective Optimization for Metrics in CLIP Image Quality Assessment
CLIP-IQA https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html

Findings:
* It is possible to optimize for multiple metrics at once
* The more criteria is optimized, the less diverse the results are (see 9 vs 3)
* The results are interesting, may not be exactly what you expect

Optimization generally done with NSGA-II and SDXL-Turbo.

## Optimizing for 9 metrics, starting with Random Embeddings
It has been shown that improvement in fitness can be made even with many criteria at the same time.

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/a0828dc2-248c-44b1-8cac-1cbabbbe238d

![NSGA II for 9 metrics](./nsga_200gen_100pop_iqavariation.png)

Parameters
```python
population_size = 100
num_generations = 200
batch_size = 1
elitism = 1
metrics = ("quality", "brightness", "colorfullness", "complexity", "natural", "happy", "new", "real", "beautiful")

embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()
creator = SDXLPromptEmbeddingImageCreator(batch_size=batch_size, inference_steps=3)
evaluator = MultiCLIPIQAEvaluator(metrics=metrics)
crossover = PooledArithmeticCrossover(crossover_rate=0.5, crossover_rate_pooled=0.5)
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=2, 
                                                     clamp_range=(embedding_range.minimum, embedding_range.maximum)) 
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=0.4, 
                                                            clamp_range=(pooled_embedding_range.minimum, pooled_embedding_range.maximum))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)

# Prepare initial arguments
init_args = [PooledPromptEmbedData(embedding_range.random_tensor_in_range(), pooled_embedding_range.random_tensor_in_range()) 
             for _ in range(population_size)] # random start

nsga = NSGA_II(
    num_generations=num_generations,
    population_size=population_size,
    solution_creator=creator,   
    selector=selector,
    crossover=crossover,
    mutator=mutator,
    evaluator=evaluator,
    elitism_count=elitism,
    initial_arguments=init_args,
    post_non_dominated_sort_callback=save_images_post_sort
)
```

[View the full notebook](./nsga_200gen_100pop_iqavariation.ipynb)

## Optimizing for Happy Sad and Quality

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/928c5fc4-2599-4ee5-a715-a4fa49d08512

Interestingly in the fitness chart it can be seen that happy and sad got closer in the end, whilst quality was able to improve throughout.

![fitness happy sad](https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/72dfcbd9-a4e9-478b-91be-691a86427e74)

```python
population_size = 50
num_generations = 100
batch_size = 1
elitism = 1
metrics = ("quality", ("Happy image.", "Sad image."), ("Sad image.", "Happy image."))

# Define min/max values for the prompt embeddings
embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()
creator = SDXLPromptEmbeddingImageCreator(batch_size=batch_size, inference_steps=4)
evaluator = MultiCLIPIQAEvaluator(metrics=metrics)
crossover = PooledArithmeticCrossover(crossover_rate=0.5, crossover_rate_pooled=0.5)
# clamp_range was evaluated with pre-testing/clamp_range/sdxl_turbo.py
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=2, 
                                                     clamp_range=(embedding_range.minimum, embedding_range.maximum)) 
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=0.4, 
                                                            clamp_range=(pooled_embedding_range.minimum, pooled_embedding_range.maximum))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)

# Prepare initial arguments
#init_embed = creator.arguments_from_prompt(prompt) # with prompt
#init_args = [init_embed for _ in range(population_size)]
init_args = [PooledPromptEmbedData(embedding_range.random_tensor_in_range(), pooled_embedding_range.random_tensor_in_range()) 
             for _ in range(population_size)] # random
```

[View the full notebook](./nsga_happysad.ipynb)

## Optimizing for Scary Beautiful Quality

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/749f9253-e35d-4eef-9372-c2e0a25ad298

![fitness scary beutiful](https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/d184494a-5950-4869-8586-435edacc1d1f)

```python
population_size = 100
num_generations = 150
batch_size = 1
elitism = 1
metrics = ("scary", "beautiful", "quality")

embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()
creator = SDXLPromptEmbeddingImageCreator(batch_size=batch_size, inference_steps=4)
evaluator = MultiCLIPIQAEvaluator(metrics=metrics)
crossover = PooledArithmeticCrossover(interpolation_weight=0.5, interpolation_weight_pooled=0.5)
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=2, 
                                                     clamp_range=(embedding_range.minimum, embedding_range.maximum)) 
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=0.4, 
                                                            clamp_range=(pooled_embedding_range.minimum, pooled_embedding_range.maximum))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = NSGATournamentSelector()

# Prepare initial arguments
#init_embed = creator.arguments_from_prompt(prompt) # with prompt
#init_args = [init_embed for _ in range(population_size)]
init_args = [PooledPromptEmbedData(embedding_range.random_tensor_in_range(), pooled_embedding_range.random_tensor_in_range()) 
             for _ in range(population_size)] # random
```

[View the full notebook](./nsga_scarybeautiful.ipynb)

## Optimizing for Complexity Color Lonely

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/cf949319-0cb0-4c98-bcd6-93bb07d1c01d

Complexity lost its best fitness over time, probably connected to lonelyness as they are in competition.

![fitness complexity lonely color](https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/f1a25a8d-fadc-4710-8a03-efc31c542125)

```python
population_size = 50
num_generations = 100
batch_size = 1
elitism = 1
metrics = ("colorfullness", "complexity", "lonely")

# Define min/max values for the prompt embeddings
embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()
creator = SDXLPromptEmbeddingImageCreator(batch_size=batch_size, inference_steps=4)
evaluator = MultiCLIPIQAEvaluator(metrics=metrics)
crossover = PooledArithmeticCrossover(crossover_rate=0.5, crossover_rate_pooled=0.5)
# clamp_range was evaluated with pre-testing/clamp_range/sdxl_turbo.py
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=2, 
                                                     clamp_range=(embedding_range.minimum, embedding_range.maximum)) 
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=0.4, 
                                                            clamp_range=(pooled_embedding_range.minimum, pooled_embedding_range.maximum))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)

# Prepare initial arguments
#init_embed = creator.arguments_from_prompt(prompt) # with prompt
#init_args = [init_embed for _ in range(population_size)]
init_args = [PooledPromptEmbedData(embedding_range.random_tensor_in_range(), pooled_embedding_range.random_tensor_in_range()) 
             for _ in range(population_size)] # random
```

[View the full notebook](./nsga_complexitylonelycolor.ipynb)
