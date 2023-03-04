
# manually tuned parameters, just for demonstration
def get_default_params(embeddings_name, clustering_name):
    params = {}
    if clustering_name == "online_csea":

        if embeddings_name == "clova":
            threshold = 0.48  # clova

        elif embeddings_name == "speechbrain":
            threshold = 0.32  # speechbrain

        elif embeddings_name == "brno":
            threshold = 0.32  # brno

        params["threshold"] = threshold

    elif clustering_name == "online_cssa":

        if embeddings_name == "clova":
            threshold = 0.34  # clova

        elif embeddings_name == "speechbrain":
            threshold = 0.19  # speechbrain

        elif embeddings_name == "brno":
            threshold = 0.17  # brno

        params["threshold"] = threshold

    elif clustering_name == "online_plda":

        if embeddings_name == "clova":
            # b, w = 0.00095, 0.00100  # clova
            threshold = 28 # clova

        elif embeddings_name == "speechbrain":
            # b, w = 0.0023, 0.0028  # speechbrain
            threshold = -18  # speechbrain
            # threshold = 5  # speechbrain

        elif embeddings_name == "brno":
            # b, w = 0.0032, 0.0045  # brno
            threshold = -18  # brno

        # params["b"] = b
        # params["w"] = w
        params["threshold"] = threshold

    elif clustering_name == "online_psda":

        if embeddings_name == "clova":
            # b, w = 307, 684  # clova
            threshold = 19  # clova

        elif embeddings_name == "speechbrain":
            # b, w = 25, 231  # speechbrain
            threshold = -21  # speechbrain

        elif embeddings_name == "brno":
            # b, w = 29, 137  # brno
            threshold = -19  # brno

        # params["b"] = b
        # params["w"] = w
        params["threshold"] = threshold

    elif clustering_name == "online_vb_plda":

        if embeddings_name == "clova":
            # b, w = 0.00095, 0.00100  # clova
            threshold = 4.3 # clova

        elif embeddings_name == "speechbrain":
            # b, w = 0.0023, 0.0028  # speechbrain
            threshold = -28 # speechbrain
            # threshold = 5  # speechbrain

        elif embeddings_name == "brno":
            # b, w = 0.0032, 0.0045  # brno
            threshold = -21.8  # brno

        # params["b"] = b
        # params["w"] = w
        params["threshold"] = threshold

    elif clustering_name == "online_vb_psda":

        if embeddings_name == "clova":
            # b, w = 307, 684  # clova
            threshold = -4.3  # clova

        elif embeddings_name == "speechbrain":
            # b, w = 25, 231  # speechbrain
            threshold = -28  # speechbrain

        elif embeddings_name == "brno":
            # b, w = 29, 137  # brno
            threshold = -18  # brno

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
