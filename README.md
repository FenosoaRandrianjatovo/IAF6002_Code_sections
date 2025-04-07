# IAF6002 Code section



<table>
  <tr>
    <td>
      <img src="https://github.com/FenosoaRandrianjatovo/IAF6002_Code_sections/blob/main/image1.png?raw=true" width="400"/>
    </td>
    <td>
      <img src="https://github.com/FenosoaRandrianjatovo/IAF6002_Code_sections/blob/main/kl.png?raw=true" width="400"/>
    </td>
  </tr>
</table>

![Image](https://github.com/FenosoaRandrianjatovo/IAF6002_Code_sections/blob/main/image1.png)



# Section 1

**L’objectif de cette [section 1](https://github.com/FenosoaRandrianjatovo/IAF6002_Code_sections/blob/main/sectioion1.py)**   est de coder les équations (1) à (6), qui servent à minimiser la divergence de Kullback-Leibler, et d’enregistrer leurs valeurs pour observer leur décroissance.

## Formulation mathématique du problème

Dans Et-SNE, la similarité entre des points en haute dimension est mesurée à l'aide de probabilités conditionnelles calculées via un noyau gaussien.

**Hypothèse du modèle gaussien**

Pour chaque point de données en haute dimension $x_i \in \mathbb{R}^d$, nous supposons que la similarité de tout autre point $x_j$ est donnée par une densité gaussienne centrée en $x_i$ avec une variance $\sigma_i^2$. Autrement dit, nous modélisons la densité de probabilité (non normalisée) comme :

$$
f(x_j \mid x_i) = \frac{1}{(2\pi\sigma_i^2)^{d/2}} \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2}\right).
$$

- **Centre :** La gaussienne est centrée en $x_i$, ce qui signifie que la similarité est basée sur la distance $\|x_i - x_j\|$.
- **Variance :** La variance $\sigma_i^2$ est spécifique à $x_i$ et est choisie via une recherche de perplexité pour refléter la densité locale autour de $x_i$.
- **Constante de normalisation :** La constante $\frac{1}{(2\pi\sigma_i^2)^{d/2}}$ s'annule lors de la formation des probabilités conditionnelles, de sorte que nous nous concentrons sur le terme exponentiel. C'est pourquoi cela ressemble à un noyau gaussien.

En utilisant cette hypothèse gaussienne, la probabilité conditionnelle que le point $x_i$ choisisse $x_j$ comme voisin est définie en normalisant le terme exponentiel sur tous les autres points :

$$
p_{j|i} = \frac{\exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2}\right)}{\sum_{k \neq i} \exp\left(-\frac{\|x_i - x_k\|^2}{2\sigma_i^2}\right)}.
$$

Pour obtenir une probabilité jointe symétrique entre $x_i$ et $x_j$, t-SNE définit

$$
p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n},
$$

On définit la similarité dans l’espace de basse dimension comme suit :

$$
q_{ij} = \frac{\left(1 + \|y_i - y_j\|^2\right)^{-1}}{\sum_{k \ne l} \left(1 + \|y_k - y_l\|^2\right)^{-1}}
$$
La fonction de perte est:
$$
C = \text{KL}(P\|Q) = \sum_{i\neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}.
$$

Pour un $p_{ij}$ fixé, on peut ignorer les termes constants.

$$
C = \sum_{i\neq j} p_{ij} \log p_{ij} -\sum_{i\neq j} p_{ij} \log q_{ij}
$$

Donc le Gradient est : 

$$
\frac{\partial C}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij})\, (y_i-y_j) (1+\|y_i-y_j\|^2)^{-1}.
$$

# Section 2

**L’objectif de cette [section 2](https://github.com/FenosoaRandrianjatovo/IAF6002_Code_sections/blob/main/section2.py)**   est de créer des fonctions permettant de charger les données synthétiques et les données réelles, en utilisant la bibliothèque `scanpy` pour le traitement des données réelles de `scVI`.

À la fin du script, j’ai également créé une fonction qui permet de reproduire les résultats en fixant les graines aléatoires (*seed*) pour `numpy`, `torch`, `cnn` et `random`.

# Section 3

**L’objectif de cette [section 3](https://github.com/FenosoaRandrianjatovo/IAF6002_Code_sections/blob/main/section3.py)**   est d’évaluer les performances de l’algorithme en réutilisant les modules implémentés dans les sections 1 et 2. Les courbes de la divergence de Kullback-Leibler ainsi que les projections en deux dimensions sont ensuite générées et sauvegardées pour une analyse visuelle.


