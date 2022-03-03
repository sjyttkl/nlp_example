## Theoretical Analysis

+ L2正则化：

​	
$$
loss=CE\left ( \widehat{y} ,y \right ) + \lambda \parallel W \parallel_{2}^{2}
$$

+ KD：

$$
loss=CE\left ( \widehat{y} ,y \right ) +\lambda loss\_soft\left ( \widehat{y} ,y \right )
$$

+  把正则化项，改为 loss_soft( 把人为的先验知识，变成老师的知识)