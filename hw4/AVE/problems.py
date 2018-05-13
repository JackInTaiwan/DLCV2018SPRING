from argparse import ArgumentParser




def tsne() :
    """
    Problem 1-5 plot t-SNE.
    """
    from sklearn.manifold.t_sne import TSNE




if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("-q", action=str, type=str, required=True)

    q = parser.parse_args().q

    if q == "tsne" :
        tsne()