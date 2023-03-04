
# manually tuned parameters, just for demonstration
def get_default_params(embeddings_name, clustering_name):
    params = {}
    if clustering_name == "online_csea":

        if embeddings_name == "clova":
            threshold = 0.47  # clova

        elif embeddings_name == "speechbrain":
            threshold = 0.28  # speechbrain

        elif embeddings_name == "brno":
            threshold = 0.29  # brno

        params["threshold"] = threshold

    elif clustering_name == "online_cssa":

        if embeddings_name == "clova":
            threshold = 0.29  # clova

        elif embeddings_name == "speechbrain":
            threshold = 0.14  # speechbrain

        elif embeddings_name == "brno":
            threshold = 0.15  # brno

        params["threshold"] = threshold

    elif clustering_name == "online_plda":

        if embeddings_name == "clova":
            # b, w = 0.00095, 0.00100  # clova
            threshold = 21.5 # clova

        elif embeddings_name == "speechbrain":
            # b, w = 0.0023, 0.0028  # speechbrain
            threshold = -30.1  # speechbrain
            # threshold = 5  # speechbrain

        elif embeddings_name == "brno":
            # b, w = 0.0032, 0.0045  # brno
            threshold = -15  # brno

        # params["b"] = b
        # params["w"] = w
        params["threshold"] = threshold

    elif clustering_name == "online_psda":

        if embeddings_name == "clova":
            # b, w = 307, 684  # clova
            threshold = -20.5  # clova

        elif embeddings_name == "speechbrain":
            # b, w = 25, 231  # speechbrain
            threshold = -34.5  # speechbrain

        elif embeddings_name == "brno":
            # b, w = 29, 137  # brno
            threshold = -17.3  # brno

        # params["b"] = b
        # params["w"] = w
        params["threshold"] = threshold

    elif clustering_name == "online_vb_plda":

        if embeddings_name == "clova":
            # b, w = 0.00095, 0.00100  # clova
            threshold = 17.3 # clova

        elif embeddings_name == "speechbrain":
            # b, w = 0.0023, 0.0028  # speechbrain
            threshold = -24.9  # speechbrain
            # threshold = 5  # speechbrain

        elif embeddings_name == "brno":
            # b, w = 0.0032, 0.0045  # brno
            threshold = -28.6  # brno

        # params["b"] = b
        # params["w"] = w
        params["threshold"] = threshold

    elif clustering_name == "online_vb_psda":

        if embeddings_name == "clova":
            # b, w = 307, 684  # clova
            threshold = 4.7  # clova

        elif embeddings_name == "speechbrain":
            # b, w = 25, 231  # speechbrain
            threshold = -33.1  # speechbrain

        elif embeddings_name == "brno":
            # b, w = 29, 137  # brno
            threshold = -22.4  # brno

        # params["b"] = b
        # params["w"] = w
        params["threshold"] = threshold

    elif clustering_name == "vb_plda":
        Fa = 0.3
        Fb = 5.0
        maxSpeakers = 30
        if embeddings_name == "clova":
            # b, w = 0.00095, 0.00100  # clova
            threshold = 0  # clova

        elif embeddings_name == "speechbrain":
            # b, w = 0.0023, 0.0028  # speechbrain
            threshold = -24.9  # speechbrain
            # threshold = 5  # speechbrain

        elif embeddings_name == "brno":
            # b, w = 0.0032, 0.0045  # brno
            threshold = -20  # brno

        # params["b"] = b
        # params["w"] = w

        params["Fa"] = Fa
        params["Fb"] = Fb
        params["maxSpeakers"] = maxSpeakers

    elif clustering_name == "vb_psda":
        Fa = 0.3
        Fb = 5.0
        maxSpeakers = 30
        if embeddings_name == "clova":
            # b, w = 307, 684  # clova
            threshold = -60  # clova

        elif embeddings_name == "speechbrain":
            # b, w = 25, 231  # speechbrain
            threshold = -33.1  # speechbrain

        elif embeddings_name == "brno":
            # b, w = 29, 137  # brno
            threshold = -20  # brno

        # params["b"] = b
        # params["w"] = w

        params["Fa"] = Fa
        params["Fb"] = Fb
        params["maxSpeakers"] = maxSpeakers

    elif clustering_name == "ahc":

        if embeddings_name == "clova":
            threshold = 0.1  # clova

        elif embeddings_name == "speechbrain":
            threshold = -0.1  # speechbrain

        elif embeddings_name == "brno":
            threshold = 0.2  # brno

        params["threshold"] = threshold

    elif clustering_name == "vb":

        maxSpeakers = 30
        loopP = 0.9
        Fa = 0.3
        Fb = 5.0

        params["maxSpeakers"] = maxSpeakers
        params["loopP"] = loopP
        params["Fa"] = Fa
        params["Fb"] = Fb

    elif clustering_name == "vb_up":

        maxSpeakers = 30
        loopP = 0.9
        Fa = 0.3
        Fb = 5.0

        params["maxSpeakers"] = maxSpeakers
        params["loopP"] = loopP
        params["Fa"] = Fa
        params["Fb"] = Fb

    return params
