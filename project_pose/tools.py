def generer_description (results):
    if not results or not results.pose_landmarks:
        return "Je ne vois personne sur cette image."

    landmarks = results.pose_landmarks.landmark
    description = []

    # Analyser la visibilité (est-ce que le corps est entier ?)
    points_visibles = sum(1 for lm in landmarks if lm.visibility > 0.5)
    description.append(f"Je détecte {points_visibles} points clés sur ton corps.")

    # Analyser l'inclinaison des épaules
    epaule_gauche = landmarks[11]
    epaule_droite = landmarks[12]
    
    if abs(epaule_gauche.y - epaule_droite.y) < 0.02:
        description.append("Tes épaules sont bien horizontales, c'est une posture stable.")
    else:
        cote = "gauche" if epaule_gauche.y < epaule_droite.y else "droit"
        description.append(f"Tu sembles penché vers le côté {cote}.")

    return " ".join(description)