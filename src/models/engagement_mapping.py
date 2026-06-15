ENGAGEMENT_MAPPING_FINAL = {
    "happy": "Engaged",
    "calm": "Engaged",
    "surprised": "Engaged",
    "angry": "Engaged",

    "neutral": "Moderately Engaged",
    "fearful": "Moderately Engaged",

    "sad": "Disengaged",
    "disgust": "Disengaged",
}


def map_emotion_to_engagement(emotion):
    return ENGAGEMENT_MAPPING_FINAL.get(emotion, "Moderately Engaged")


def get_logic_based_feedback(engagement_level):
    if engagement_level == "Engaged":
        return (
            "Overall classroom engagement is strong. Students appear attentive or actively involved. "
            "Continue discussion-based teaching and use follow-up legal reasoning questions "
            "to sustain participation."
        )

    if engagement_level == "Moderately Engaged":
        return (
            "Overall classroom engagement is moderate. Introduce an interactive activity such as "
            "a short debate prompt, case-law question, quick student response, or practical legal "
            "scenario to increase participation."
        )

    if engagement_level == "Disengaged":
        return (
            "Students show possible low-engagement signals. Use an interactive activity such as "
            "role-play, debate prompt, moot-court style question, or practical legal scenario "
            "to regain attention."
        )

    return (
        "Engagement could not be clearly determined. Review the classroom recording quality "
        "and repeat the analysis if needed."
    )


def get_feedback_for_engagement(engagement_level):
    return get_logic_based_feedback(engagement_level)