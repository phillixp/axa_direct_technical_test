def target_encoding(train, serie='origine', target_name='y_train', k=None, f=None, avec_viz=True, noise_level=0):
    """
    Encode une variable catégorielle non ordinale à haute cardinalité à l'aide de la distribution CONTINUE ou BINAIRE de la @target.
    Le principe est le même que pour celui de la crédibilité : on fait "confiance" à une
    modalité Xi quand elle contient suffisamment d'observations : la fonction de blending-confiance λ(Ni) prendra des
    valeurs proches de 1, et l'encoding s'approchera de E[Y | X=Xi].
    À l'inverse, si une modalité Xi n'est prise que par quelques observations, on préfèrera faire confiance
    à la statistique globale E[Y] sur tout le train.

    ---params---
    @k : paramètre de position : à partir de quel effectif de modalité on commence à faire davantage confiance au posterior qu'au prior.
    @f : paramètre d'inflexion, provides control over the slope of the function around the inflexion point @k.
    @avec_viz : si True, calcule la sigmoid
    @noise_level : in [0, 1], bruit aléatoire, un pourcentage de l'étendue de la target à ajouter aux valeurs. Si > 0, retourne
    la colonne à ajouter après le mapping à la série encodée (en plus du mapping) --> "return mapping, noise_level"

    Usage :
    > encoding, mapping = target_encoding(mini_train, serie='origine', target_name='label', k=150, f=40, avec_viz=True)
    > encoding, mapping, noise = target_encoding(mini_train, serie='origine', target_name='label', k=150, f=40, avec_viz=True, noise=0.01)
    > mini_train['origine_TE'] = encoding + noise
    > valid['origine_TE'] = valid['origine'].map(mapping) # surtout pas de bruit ici !
    > valid['origine_TE'].fillna(mapping[np.nan])  # de nouvelles modalités peuvent apparaître dans le valid
    > test['origine_TE'] = test['origine'].map(mapping) # surtout pas de bruit ici !
    > test['origine_TE'].fillna(mapping[np.nan])  # de nouvelles modalités peuvent apparaître dans le test

    Voir http://helios.mm.di.uoa.gr/~rouvas/ssi/sigkdd/sigkdd.vol3.1/barreca.pdf
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1 - calcul du prior P(Y)
    prior = train[target_name].mean()

    # 2 - calcul du posterior P(Y | X)
    #   previous dans la V1 : posterior = train.groupby(serie)[target_name].mean(), calculé en utilisant la target (pas bien)
    #   à présent : moyenne calculée sans utiliser la ligne en cours
    s, c = train.groupby(serie)[target_name].transform('sum'), train.groupby(serie)[target_name].transform('count')
    map_table = pd.DataFrame({'Y':train[target_name], 'Somme':s, 'Count':c})
    map_table['CountMoins1'] = map_table['Count'] - 1
    map_table['SumSurMoins1'] = (map_table['Somme'] - map_table['Y']) 
    map_table['MoyLeaveOneOut'] = map_table['SumSurMoins1'] / (map_table['CountMoins1'])
    posterior = map_table['MoyLeaveOneOut'].fillna(prior)

    # 3 - calcul de la fonction de blending
    def confiance(Ni, k, f):
        import numpy as np
        """
        La fonction de blending paramétrée par k, f, et Ni (nb de lignes pour lesquelles X = Xi).
        Retourne le niveau de confiance à accorder au posterior (entre 0 et 1) : sur un budget de 100 points de confiance,
        combien en alloue t-on au prior, et combien au posterior ?
        @k : The position parameter k determines half of the minimal sample size for which we completely trust the estimate based on the sample in the cell.
        @f : The inflexion parameter f provides control over the slope of the function around the inflexion point.
        """
        return 1 / (1 + np.exp(-(Ni-k)/f))

    # Par défaut et sauf avis contraire de l'utilisateur, on commence à faire davantage confiance au
    #   posterior P(Y | Xi) qu'au prior quand Ni dépasse 100 lignes (previous : dépasse 5% de l'effectif total) :
    if k is None : k=100  # previous : 0.05*len(train)
    if f is None : f=1000

    if avec_viz:
        liste = [confiance(Ni, k, f) for Ni in np.arange(1, len(train))] # on calcule tous les 100 la valeurs de confiance
        plt.plot(liste)
        plt.scatter([k], [0.51], color='r', s=100, label=f'k = {k:.0f}', zorder=6, marker='x')

        # Calcul de la tangeante :
        Ni = len(train)*(k/len(train))
        a = (confiance(Ni, k-1, f) - confiance(Ni, k+1, f))/2
        b = confiance(Ni, k-1, f) - (k-1)*a
        plt.plot([0, len(train)], [a*0+b, a*len(train)+b], label=f'f = {f}', linewidth=3)

        plt.xlabel(r"$N_i$ : " + "l'effectif de la modalité\n d'une variable", fontsize=12)
        plt.ylabel(r"$\lambda(N_i)$" + " : la confiance\naccordée au posterior", fontsize=12)
        plt.legend(fontsize=10)
        plt.ylim((-1e-1, 1+1e-1)) ; plt.xlim((0, len(train)))
        plt.title("Quelle confiance accorder à une modalité " +  r"représentée $N_i$ fois ?", fontsize=17, y=1.05)
        
    confiance = dict(confiance(train[serie].value_counts(), k, f))
    confiance_serie = train[serie].map(confiance).fillna(0)

    # Création du mapping servant à encoder le valid et le test :
    encoding = pd.DataFrame({'Modalité':train[serie], 
                             "Encoding":confiance_serie*posterior + (1-confiance_serie)*prior})
    mapping = dict(encoding.groupby('Modalité').Encoding.mean())
    mapping.update({np.nan:prior})

    if noise_level > 0 :
        target_étendue = train[target_name].max() - train[target_name].min()
        noise_level = np.random.rand(len(train)) * noise_level * target_étendue
        return confiance_serie*posterior + (1-confiance_serie)*prior, mapping, noise_level

    return confiance_serie*posterior + (1-confiance_serie)*prior, mapping
