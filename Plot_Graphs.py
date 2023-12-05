import matplotlib.pyplot as plt



def graph_loss(y_values, figure_name):
    x_values = [int(i) for i in range(len(y_values[0][0]))]
    for i in range(len(y_values)):
        plt.plot(x_values, y_values[i][0], label = y_values[i][1])

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.savefig(figure_name)
    plt.clf()
    return 


def graph_accuracies(y_values, figure_name):
    x_values = [i for i in range(len(y_values[0][0]))]
    for i in range(len(y_values)):
        plt.plot(x_values, y_values[i][0], label = y_values[i][1])
        
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(figure_name)
    plt.clf()
    return


def main():
    #Baseline
    Baseline_loss = [0.6890, 0.6265, 0.6023, 0.5877, 0.5514, 0.5581, 0.5667, 0.5708, 0.5545, 0.5636, 0.5657, 0.5383, 0.5369, 0.5355, 0.5531, 0.5192, 0.5358, 0.5353, 0.5601, 0.5535]
    Baseline_accuracies = [0.5600, 0.6550, 0.6650, 0.7050, 0.7500, 0.7200, 0.7250, 0.7250, 0.7550, 0.7437, 0.7200, 0.7350, 0.7550, 0.7850, 0.7150, 0.7700, 0.7600, 0.7400, 0.7300, 0.7550]

    #Add Bayesian Optimization
    Bayesian_loss = []
    Bayesian_accuracies = []

    #GAN Model
    gan_loss = [0.0192, 0.1329, 0.1407, 0.0274, 0.0273, 0.0342, 0.0248, 0.0228, 0.0081, 0.0278, 0.0258, 0.0257, 0.0424, 0.0203, 0.0167, 0.0218, 0.0189, 0.0255, 0.0090, 0.0082]
    gan_accuracies = [242/600, 246/600, 251/600, 265/600, 283/600, 291/600, 317/600, 292/600, 324/600, 338/600, 323/600, 345/600, 354/600, 356/600, 359/600, 360/600, 366/600, 366/600, 395/600, 409/600]

    graph_loss([[Baseline_loss, "Baseline"], [gan_loss, "GAN"]], "Training_loss.png")
    graph_accuracies([[Baseline_accuracies, "Baseline"], [gan_accuracies, "GAN"]], "Training_accuracies.png")


main()