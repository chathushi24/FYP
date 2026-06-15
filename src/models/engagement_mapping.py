ENGAGEMENT_MAPPING_FINAL = {
    "happy": "Engaged",
    "calm": "Engaged",
    "surprised": "Engaged",
    "angry": "Moderately Engaged",

    "neutral": "Moderately Engaged",
    "fearful": "Moderately Engaged",

    "sad": "Disengaged",
    "disgust": "Disengaged",
}


def map_emotion_to_engagement(emotion: str) -> str:
    return ENGAGEMENT_MAPPING_FINAL.get(str(emotion).lower(), "Moderately Engaged")


def get_feedback_for_engagement(level: str) -> str:
    if level == "Engaged":
        return (
            "Students appear engaged. Continue the discussion and introduce deeper "
            "case-based legal reasoning questions."
        )

    if level == "Moderately Engaged":
        return (
            "Engagement appears moderate or at-risk. Use a quick question, case prompt, "
            "or short debate activity to clarify understanding and increase participation."
        )

    return (
        "Students show possible low-engagement signals. Use an interactive activity "
        "such as role-play, debate prompt, or practical legal scenario."
    )