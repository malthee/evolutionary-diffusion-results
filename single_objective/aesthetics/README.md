# Aesthetic Optimization Results
The Aesthetics Evaluator is based on the LAION Aesthetics Predictor V2. Source: https://laion.ai/blog/laion-aesthetics/ and GitHub https://github.com/christophschuhmann/improved-aesthetic-predictor. 

Findings: 
* Using a greater population size is recommended to allow for variation in the population. Experiments with a lower population size and less diversity converged very quickly.
* Mutation should be restricted to a reasonable CLAMP range. When these values are exceeded the generational model produces weird results. 
* Arithmetic Crossover works well with default configuration and 0.5 crossover rate.
* Elitism helped in exploitation, otherwise good results would have gotten lost 
* Optimization for a speficic prompt works well when reducing the initial prompt search space and reducing mutation. May still lead to loss of some prompt components.

## Optimizing a GA for Maximum Aesthetics with SDXL Turbo
Optimizing the aesthetics predictor as a maximization problem, the algorithm came to a max Aesthetics score of **8.67**.
This score is higher than [the examples from the real LAION English Subset dataset have](http://captions.christoph-schuhmann.de/aesthetic_viz_laion_sac+logos+ava1-l14-linearMSE-en-2.37B.html).
A wide variety of prompts (inspired by parti prompts) was used for the initial population.

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/4841d671-639a-4ac4-b7a8-ee5a66fab28d

![Ga200Gen100PopFitnessChartAesthetics](./ga_200gen_100pop_aesthetic.png)

Parameters: 
```python
population_size = 100
num_generations = 200
batch_size = 1
elitism = 1

creator = SDXLPromptEmbeddingImageCreator(pipeline_factory=setup_pipeline, batch_size=batch_size, inference_steps=3)
evaluator = AestheticsImageEvaluator()  
crossover = PooledArithmeticCrossover(crossover_rate=0.5, crossover_rate_pooled=0.5)
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=2, clamp_range=(-900, 900)) 
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=0.3, clamp_range=(-8, 8))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)
```

[View the full notebook](./ga_200gen_100pop_aesthetic.ipynb)

## Optimizing a GA for Minimum Aesthetics with SDXL Turbo
Optimizing the aesthetics predictor as a minimization problem, the algorithm came to a min Aesthetics score of **0.457**. Similar to above
this was able to beat any real image from the dataset.

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/4352cdc0-20d6-4547-864d-e174f52204f3

![Ga200Gen100PopFitnessChartInvaesthetics](./ga_200gen_100pop_invaesthetic.png)

Parameters:
```python
population_size = 100
num_generations = 200
batch_size = 1
elitism = 1

creator = SDXLPromptEmbeddingImageCreator(pipeline_factory=setup_pipeline, batch_size=batch_size, inference_steps=3)
evaluator = InverseEvaluator(AestheticsImageEvaluator()) # Inverting aesthetics to try getting the worst image
crossover = PooledArithmeticCrossover(crossover_rate=0.5, crossover_rate_pooled=0.5)
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=2, clamp_range=(-900, 900)) 
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=0.3, clamp_range=(-8, 8))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)
```

[View the full notebook](./ga_200gen_100pop_invaesthetic.ipynb)

## Optimizing Aesthetics for a specific Prompt
Another use case is to improve aesthetics for a specific prompt. This can be done by starting in a restricted search space - such as one defined by 99% by the prompt and 1% random and then search from there on.

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/17f09800-3346-4eb9-9834-e71f3f29b250

![prompt specific aesthetics improvement](https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/4319d6d0-ec0b-44d0-8ccc-82ad291d9ddc)

```python
population_size = 50
num_generations = 100
batch_size = 1
elitism = 1
inference_steps = 4
crossover_proportion = 0.8
crossover_rate = 0.9
mutation_rate = 0.2
strict_osga = False
prompt = "a cozy flat with a fireplace, a dog and cat sleeping on the couch"

# Define min/max values for the prompt embeddings
embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()

creator = SDXLPromptEmbeddingImageCreator(batch_size=batch_size, inference_steps=inference_steps)
evaluator = AestheticsImageEvaluator()
crossover = PooledArithmeticCrossover(interpolation_weight=0.5, interpolation_weight_pooled=0.5, 
                                      proportion=crossover_proportion, proportion_pooled=crossover_proportion)
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.05, mutation_strength=1, 
                                                     clamp_range=(embedding_range.minimum, embedding_range.maximum)) 
mutation_arguments_pooled = UniformGaussianMutatorArguments(mutation_rate=0.05, mutation_strength=0.2, 
                                                            clamp_range=(pooled_embedding_range.minimum, pooled_embedding_range.maximum))
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)

# Prepare initial arguments, random population of *reasonable* prompt embeddings
init_args = [PooledPromptEmbedData(embedding_range.random_tensor_in_range(), pooled_embedding_range.random_tensor_in_range()) 
                 for _ in range(population_size)]
init_crossover = PooledArithmeticCrossover(0.99, 0.99)
prompt_args = creator.arguments_from_prompt(prompt)
init_args = [init_crossover.crossover(prompt_args, args) for args in init_args]
```

[View the full notebook](./ga_aesthetics_restricted.ipynb)

## Optimizing a GA for Maximum Aesthetics with SD Turbo 
Trying out a similar experiment with comparable parameters switching out SDXL for SD Turbo. This resulted in a final score of **7.9**, which is lower than its SDXL variant. 

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/549f3536-1b2f-4732-9e45-0f6a57ebd34c

![SDGa200Gen100PopFitnessChartAesthetics](./sd_ga_200gen_100pop_aesthetic.png)

Parameters
```python
population_size = 100
num_generations = 200
batch_size = 1
elitism = 1
inference_steps = 3

creator = SDPromptEmbeddingImageCreator(pipeline_factory=setup_pipeline, batch_size=batch_size, inference_steps=inference_steps)
evaluator = AestheticsImageEvaluator() 
crossover = ArithmeticCrossover(0.5)
mutation_arguments = UniformGaussianMutatorArguments(mutation_rate=0.1, mutation_strength=0.4, clamp_range=(-10.3, 15.65)) 
mutator = UniformGaussianMutator(mutation_arguments)
selector = TournamentSelector(tournament_size=3)
```


[View the full notebook](./sd_ga_200gen_100pop_aesthetic.ipynb)

## OSGA Random Optimization
A strict OSGA approach was tried out, so that only better offspring are accepted. This lead to similar results than normal GA with more time invested, but less generations. (38 Hours, Fitness just over 8, 11 Generations)

https://github.com/malthee/evolutionary-diffusion-results/assets/18032233/fdddd57d-bec0-4a78-9234-4bcc9e5f99bb

[View the full notebook](./osga_aesthetics.ipynb)
