
This project looks at the persistent homology of arrays that expand in a streaming fashion, meaning arriving chunks keep incrementing an array, hence the name **Pe**rsistence **Ch**unks **Stre**aming.

One way to compute an array's persistence is to re-compute it for the entire array every time a new chunk arrives. While this works fine, it does unnecessary work, especially for massive data like maps or videos.

The filtration method used to process the birth and death of the data depends on the modality. For instance:
- 2D arrays, like images, commonly use sublevel/superlevel set filtration based on pixel intensity or a scalar function
- point clouds use distance-based filtration like Vietoris-Rips, Alpha/Delaunay, Čech, or witness complexes
- time series often do sublevel set on the signal, or alternately, undergo time delay (Takens) embedding to turn into a point cloud, after which distance-based filtrations are applied
- Graphs/networks mostly use weight filtrations on edges or nodes

Our method focuses on the sublevel set filtration and uses a cubical complex.

