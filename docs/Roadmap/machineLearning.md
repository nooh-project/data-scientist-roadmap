## Fundamentos

Sendo baseado em técnicas matemáticas e estatísticas, o Machine learning acaba sendo a interseção de duas disciplinas: ciência de dados e engenharia de software. O objetivo do machine learning é usar dados históricos, de observações passadas, para criar um modelo preditivo que pode ser incorporado em uma aplicação ou serviço de software

Fundamentalmente, um modelo de machine learning é uma aplicação de software que encapsula uma função para calcular um valor de saída com base em um ou mais valores de entrada. O processo de definição dessa função é conhecido como treinamento. Após a função ser definida, ela pode ser usada para prever novos valores em um processo chamado inferência

O processo de treinamento de um modelo de machine learning envolve a utilização de dados históricos, conhecidos como dados de treinamento. Esses dados são compostos por observações passadas que incluem, na maioria dos casos, duas partes principais:

- **Features (Atributos ou Características)**: São os fatores observáveis ou medíveis do objeto ou evento em questão. Eles representam as variáveis independentes que o modelo utilizará para fazer previsões. Por exemplo, se você está treinando um modelo para prever o preço de casas, as características podem incluir a metragem quadrada, o número de quartos, a localização, etc. Geralmente associados a variável *X*

- **Label (Rótulo)**: É o valor conhecido daquilo que você quer que o modelo preveja. Também pode ser chamado de variável dependente ou variável de destino. No exemplo do preço das casas, o rótulo seria o preço de venda conhecido de cada casa. Geralmente associados a variável *y*

No treinamento do modelo, um algoritmo é então aplicado aos dados com o objetivo de determinar uma relação entre as features e o label, e generalizar essa relação como um cálculo que pode ser realizado sobre X para calcular y. O algoritmo específico utilizado depende do tipo de problema preditivo que você está tentando resolver (mais sobre isso depois), mas o princípio básico é tentar ajustar uma função aos dados, na qual os valores das características podem ser usados para calcular o rótulo

Quando o algoritmo é aplicado aos dados, o resultado é um modelo que encapsula o cálculo derivado pelo algoritmo como uma função *f*, sendo matematicamente representado como: *y = f(X)*

Com a fase de treinamento concluída, o modelo treinado pode ser utilizado para inferência. O modelo é essencialmente um programa de software que encapsula a função produzida pelo processo de treinamento. Podendo ser inserido um conjunto de valores de características e receber como saída uma previsão do rótulo correspondente. Como a saída do modelo é uma previsão calculada pela função, e não um valor observado, você frequentemente verá a saída da função mostrada como ŷ (pronunciado como "y-hat")

[![Diagrama de Machine Learning](./assets/machineLearningFlow.png)](https://learn.microsoft.com/en-us/training/modules/fundamentals-machine-learning/2-what-is-machine-learning)

<br>

## Tipos de ML

[![Tipos de Machine Learning](./assets/machineLearningTypes.png)](https://learn.microsoft.com/en-us/training/modules/fundamentals-machine-learning/3-types-of-machine-learning)

