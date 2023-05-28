# Classificação de Imagens usando Deep Learning
Este repositório contém um exemplo de como treinar e usar um modelo de aprendizado profundo para classificação de imagens usando a biblioteca TensorFlow. O modelo é treinado em um conjunto de dados contendo imagens de duas classes diferentes.

Pré-requisitos
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

Conjunto de Dados:

O conjunto de dados usado neste exemplo contém imagens de duas classes: "cachorro" e "cavalo". As imagens estão organizadas em uma estrutura de diretórios onde cada diretório corresponde a uma classe. Certifique-se de organizar seus dados de maneira semelhante antes de treinar o modelo.

Treinamento do Modelo

Configure as variáveis no início do script para corresponderem às suas configurações, como o diretório de dados, altura e largura das imagens, número de classes, número de épocas, etc.

Execute o script para treinar o modelo. Durante o treinamento, a acurácia e a perda do treinamento e da validação serão exibidas. O modelo treinado será salvo em um arquivo chamado "meu_modelo.h5".

Uso do Modelo Treinado

Carregue o modelo treinado a partir do arquivo "meu_modelo.h5".

Use a função predict_image para prever a classe de uma única imagem. Passe o caminho da imagem, altura e largura desejadas como argumentos. A função retornará uma matriz de probabilidades para cada classe.

Use a função get_class_label para obter o rótulo da classe com base nas probabilidades retornadas pela função predict_image.

Exemplo de Classificação de Imagens

Defina os caminhos das imagens que você deseja classificar na lista image_paths.

Execute o código para carregar o modelo treinado, prever as classes das imagens e exibir as imagens com os rótulos e probabilidades correspondentes.

Gráficos de Acurácia e Perda

Os gráficos de acurácia e perda do treinamento e da validação serão exibidos após o treinamento do modelo. Esses gráficos ajudam a visualizar o desempenho do modelo ao longo das épocas.

Interpretação dos Resultados

Os resultados do último epoch de treinamento serão exibidos, incluindo a acurácia do treinamento e da validação, bem como a perda do treinamento e da validação. O script também fornecerá interpretações gerais dos resultados, como indicação de overfitting ou acurácia razoável na validação.

Observação: Certifique-se de ajustar as configurações e os caminhos do diretório de dados de acordo com seu próprio cenário antes de executar o código.


Espero que este exemplo seja útil para entender como treinar e usar um modelo de classificação de imagens usando TensorFlow. Sinta-se à vontade para personalizar e adaptar o código conforme necessário para atender às suas necessidades específicas.
