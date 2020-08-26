# Regularize image embedding with spatial continuity prior

Location-aware image clustering.

Clustering image patches of a giant histopathological image allows highlighting of regions of similar texture, with applications like image retrieval [1] and image segmentation [2-3]. Common practices of image clustering and classification are solely based on image content, and hence treating each patch independently. In certain cases, patches next to each other are more likely to have same or similar labels, reflecting the spatial continuity nature of the label. Such information can be introduced to image clustering through methods like supervised dimension reduction [4], providing extra information and therefore better clustering/classification quality.

![wsi](figures/wsi.png)
![wsi_label](figures/wsi_label.png)
![cluster](figures/plot_cluster.png)
![metric_chart](figures/metric_chart.png)

## Reference
[1] Barcode Annotations for Medical Image Retrieval: A Preliminary Investigation ([arxiv](https://arxiv.org/abs/1505.05212))  
[2] Atlas of Digital Pathology: A Generalized Hierarchical Histological Tissue Type-Annotated Database for Deep Learning ([DOI](https://doi.org/10.1109/CVPR.2019.01202))  
[3] HistoSegNet: Semantic Segmentation of Histological Tissue Type in Whole Slide Images ([DOI](https://doi.org/10.1109/ICCV.2019.01076))  
[4] UMAP for Supervised Dimension Reduction and Metric Learning ([link](https://umap-learn.readthedocs.io/en/latest/supervised.html))
