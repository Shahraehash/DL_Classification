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
    Baseline_loss = [0.44038, 0.45818, 0.52254, 0.43997, 0.44880, 0.38413, 0.40484, 0.41656, 0.43872, 0.52904, 0.46199, 0.44494, 0.44299, 0.42713, 0.39541, 0.48086, 0.57322, 0.45233, 0.33279, 0.39548]
    Baseline_accuracies = [0.5750, 0.5750, 0.5450, 0.5650, 0.5800, 0.6200, 0.6050, 0.6050, 0.6150, 0.5300, 0.6100, 0.6150, 0.5250, 0.6250, 0.6550, 0.5800, 0.5400, 0.6150, 0.5850, 0.6400]

    #Add Bayesian Optimization

    #GAN Model
    gan_loss = [0.0199, 0.0151, 0.0168, 0.0229, 0.0162, 0.0136, 0.0262, 0.0236, 0.0169, 0.0214, 0.0411, 0.0402, 0.0189, 0.0184, 0.0398, 0.0271, 0.0310, 0.0380, 0.0268, 0.0097]
    gan_accuracies = [341/600, 368/600, 361/600, 366/600, 379/600, 371/600, 331/600, 366/600, 344/600, 294/600, 337/600, 360/600, 280/600, 302/600, 291/600, 311/600, 304/600, 332/600, 382/600, 386/600]


    graph_loss([[Baseline_loss, "Baseline"], [gan_loss, "GAN"]], "Training_loss.png")
    graph_accuracies([[Baseline_accuracies, "Baseline"], [gan_accuracies, "GAN"]], "Training_accuracies.png")


main()