# knn-search

Python + Rust app for kNN search. `main.py` example uses `BAAI/bge-large-en-v1.5` model to encode `rajpurkar/squad` dataset, then finds the most similar contexts using cosine similarity, compares Rust and Python performance.

## Setup

```bash
uv venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

uv sync
maturin develop --release
```

## Run

```bash
python main.py
```


## Example output

```
Unique contexts: 18891
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 15284.13it/s]
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 591/591 [00:59<00:00,  9.88it/s]
Query: How many square miles is Oklahoma City?

Top 5 similar contexts (Rust):
- similarity: 0.8594 | The population density normally reported for Oklahoma City using the area of its city limits can be a bit misleading. Its urbanized zone covers roughly 244 sq mi (630 km2) resulting in a density of 2,500 per square mile (2013 est), compared with larger rural watershed areas incorporated by the city, which cover the remaining 377 sq mi (980 km2) of the city limits.
- similarity: 0.7760 | Oklahoma City is the capital and largest city of the state of Oklahoma. The county seat of Oklahoma County, the city ranks 27th among United States cities in population. The population grew following the 2010 Census, with the population estimated to have increased to 620,602 as of July 2014. As of 2014, the Oklahoma City metropolitan area had a population of 1,322,429, and the Oklahoma City-Shawnee Combined Statistical Area had a population of 1,459,758 (Chamber of Commerce) residents, making it Oklahoma's largest metropolitan area. Oklahoma City's city limits extend into Canadian, Cleveland, and Pottawatomie counties, though much of those areas outside of the core Oklahoma County area are suburban or rural (watershed). The city ranks as the eighth-largest city in the United States by land area (including consolidated city-counties; it is the largest city in the United States by land area whose government is not consolidated with that of a county or borough).
- similarity: 0.7671 | According to the United States Census Bureau, the city had a total area of 17.037 square miles (44.125 km2), including 10.747 square miles (27.835 km2) of land and 6.290 square miles (16.290 km2) of water (36.92%).
- similarity: 0.7664 | According to the United States Census Bureau, the city has a total area of 620.34 square miles (1,606.7 km2), of which, 601.11 square miles (1,556.9 km2) of it is land and 19.23 square miles (49.8 km2) of it is water. The total area is 3.09 percent water.
- similarity: 0.7565 | Oklahoma had 598 incorporated places in 2010, including four cities over 100,000 in population and 43 over 10,000. Two of the fifty largest cities in the United States are located in Oklahoma, Oklahoma City and Tulsa, and 65 percent of Oklahomans live within their metropolitan areas, or spheres of economic and social influence defined by the United States Census Bureau as a metropolitan statistical area. Oklahoma City, the state's capital and largest city, had the largest metropolitan area in the state in 2010, with 1,252,987 people, and the metropolitan area of Tulsa had 937,478 residents. Between 2000 and 2010, the cities that led the state in population growth were Blanchard (172.4%), Elgin (78.2%), Jenks (77.0%), Piedmont (56.7%), Bixby (56.6%), and Owasso (56.3%).

Top 5 similar contexts (Python):
- similarity: 0.8594 | The population density normally reported for Oklahoma City using the area of its city limits can be a bit misleading. Its urbanized zone covers roughly 244 sq mi (630 km2) resulting in a density of 2,500 per square mile (2013 est), compared with larger rural watershed areas incorporated by the city, which cover the remaining 377 sq mi (980 km2) of the city limits.
- similarity: 0.7760 | Oklahoma City is the capital and largest city of the state of Oklahoma. The county seat of Oklahoma County, the city ranks 27th among United States cities in population. The population grew following the 2010 Census, with the population estimated to have increased to 620,602 as of July 2014. As of 2014, the Oklahoma City metropolitan area had a population of 1,322,429, and the Oklahoma City-Shawnee Combined Statistical Area had a population of 1,459,758 (Chamber of Commerce) residents, making it Oklahoma's largest metropolitan area. Oklahoma City's city limits extend into Canadian, Cleveland, and Pottawatomie counties, though much of those areas outside of the core Oklahoma County area are suburban or rural (watershed). The city ranks as the eighth-largest city in the United States by land area (including consolidated city-counties; it is the largest city in the United States by land area whose government is not consolidated with that of a county or borough).
- similarity: 0.7671 | According to the United States Census Bureau, the city had a total area of 17.037 square miles (44.125 km2), including 10.747 square miles (27.835 km2) of land and 6.290 square miles (16.290 km2) of water (36.92%).
- similarity: 0.7664 | According to the United States Census Bureau, the city has a total area of 620.34 square miles (1,606.7 km2), of which, 601.11 square miles (1,556.9 km2) of it is land and 19.23 square miles (49.8 km2) of it is water. The total area is 3.09 percent water.
- similarity: 0.7565 | Oklahoma had 598 incorporated places in 2010, including four cities over 100,000 in population and 43 over 10,000. Two of the fifty largest cities in the United States are located in Oklahoma, Oklahoma City and Tulsa, and 65 percent of Oklahomans live within their metropolitan areas, or spheres of economic and social influence defined by the United States Census Bureau as a metropolitan statistical area. Oklahoma City, the state's capital and largest city, had the largest metropolitan area in the state in 2010, with 1,252,987 people, and the metropolitan area of Tulsa had 937,478 residents. Between 2000 and 2010, the cities that led the state in population growth were Blanchard (172.4%), Elgin (78.2%), Jenks (77.0%), Piedmont (56.7%), Bixby (56.6%), and Owasso (56.3%).

Rust time: 0.1851s
Python time: 1.5903s
Rust is 8.6x faster
```