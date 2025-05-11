# webcam_feed.py

import cv2
import tempfile
import streamlit as st
import time
import os

from backend.pose_detection.mediapipe_model import PoseDetector
from backend.feedback_engine.pose_comparator import (
    load_single_reference_landmarks,
    compute_pose_accuracy,
    check_enough_landmarks
)
from backend.feedback_engine.motion_tools import load_motion_reference, compute_motion_similarity
from backend.voice.voice_feedback_clips.tts_engine import speak

from backend.feedback_engine.yoga_feedback_engine import (
    get_feedback_tags,
    get_feedback_tags_vrikshasana,
    feedback_lines
)
from database.logger import init_db, log_session


def run_pose_detection(pose_name="tadasana", category="Yoga & Meditation"):
    init_db()

    st.session_state.setdefault("running", False)
    st.session_state.setdefault("start_time", None)
    st.session_state.setdefault("feedback_collected", set())
    st.session_state.setdefault("stop", False)
    st.session_state.setdefault("reps", 0)
    st.session_state.setdefault("pose_held", False)

    detector = PoseDetector()
    coach = speak

    reference_landmarks = None
    motion_reference = None

    if category == "Yoga & Meditation":
        try:
            corrected_pose_name = pose_name.capitalize()
            reference_landmarks = load_single_reference_landmarks(f"pose_references/{corrected_pose_name}.npz")
        except Exception:
            st.error(f"❌ Could not load reference for {pose_name}.")
            return
    elif category == "Workout & Training":
        motion_reference = load_motion_reference(f"motion_references/{pose_name}_motion.npz")
        if motion_reference is None:
            st.error(f"❌ No motion reference found for {pose_name}.")
            return

    if not st.session_state.running:
        st.session_state.running = True
        st.session_state.start_time = time.time()
        st.session_state.feedback_collected = set()
        st.session_state.stop = False
        st.session_state.reps = 0
        st.session_state.pose_held = False
        st.success("✅ Session started. Your form will now be monitored...")
        coach("start_session")

    process_camera(pose_name, detector, coach, reference_landmarks, motion_reference)


def process_camera(pose_name, detector, coach, reference_landmarks, motion_reference):
    camera = cv2.VideoCapture(0)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    stframe = st.empty()
    feedback_placeholder = st.empty()
    accuracy_display = st.empty()
    reps_display = st.empty()

    # Only show stop button when running
    if st.session_state.running:
        st.button("🔚 Stop Session", key="stop_button", on_click=lambda: st.session_state.update({"stop": True}))

    last_accuracy = 0
    motion_buffer = []
    spoken_tags = set()
    last_feedback = None

    while camera.isOpened():
        if st.session_state.stop:
            camera.release()
            break

        ret, frame = camera.read()
        if not ret:
            st.error("❌ Camera error.")
            break

        frame = cv2.flip(frame, 1)

        results = detector.detect_pose(frame)
        frame = detector.draw_landmarks(frame, results)
        named_landmarks = detector.get_named_landmarks(results)
        raw_landmarks = detector.get_landmarks(results)

        if named_landmarks and category_is_yoga(pose_name):
            if pose_name == "tadasana":
                tags = get_feedback_tags(named_landmarks)
            elif pose_name == "vrikshasana":
                tags = get_feedback_tags_vrikshasana(named_landmarks)
            else:
                tags = []

            print("🧠 Detected Tags:", tags)  # debug

            for tag in tags:
                if tag not in spoken_tags:
                    coach(tag)
                    spoken_tags.add(tag)
                    st.session_state.feedback_collected.add(tag)
                    last_feedback = tag
                    time.sleep(0.5)

            if "pose_correct" in tags:
                spoken_tags.clear()

        elif raw_landmarks and motion_reference is not None:
            if not check_enough_landmarks(raw_landmarks):
                feedback_placeholder.warning("⚠️ Pose not fully visible.")
                accuracy_display.metric("🎯 Accuracy", "0%")
                if last_feedback != "pose_not_visible":
                    coach("pose_not_visible")
                    last_feedback = "pose_not_visible"
            else:
                motion_buffer.append(raw_landmarks)
                if len(motion_buffer) > 10:
                    motion_buffer = motion_buffer[-10:]

                accuracy = compute_motion_similarity(motion_buffer, motion_reference)
                last_accuracy = (0.7 * last_accuracy) + (0.3 * accuracy)
                accuracy_display.metric("🎯 Accuracy", f"{last_accuracy:.2f}%")
                reps_display.metric("✅ Reps", st.session_state.reps)

                if accuracy > 80 and not st.session_state.pose_held:
                    st.session_state.reps += 1
                    st.session_state.pose_held = True
                    coach("great_rep")

                if accuracy < 40:
                    st.session_state.pose_held = False

                if accuracy < 60:
                    feedback_placeholder.markdown("### ⚠️ Adjust your form!")
                    coach("adjust_form")
                else:
                    feedback_placeholder.markdown("### ✅ Looking good!")

        elif named_landmarks and reference_landmarks is not None:
            if not check_enough_landmarks(raw_landmarks):
                feedback_placeholder.warning("⚠️ Pose not fully visible.")
                accuracy_display.metric("🎯 Accuracy", "0%")
                if last_feedback != "pose_not_visible":
                    coach("pose_not_visible")
                    last_feedback = "pose_not_visible"
            else:
                accuracy = compute_pose_accuracy(raw_landmarks, reference_landmarks)
                last_accuracy = (0.7 * last_accuracy) + (0.3 * accuracy)
                accuracy_display.metric("🎯 Accuracy", f"{last_accuracy:.2f}%")

                if last_accuracy >= 90 and last_feedback != "pose_correct":
                    feedback_placeholder.success("✅ Excellent posture!")
                    coach("pose_correct")
                    last_feedback = "pose_correct"

                elif last_accuracy >= 75 and last_feedback != "minor_correction":
                    feedback_placeholder.warning("⚠️ Minor Adjustments Needed!")
                    coach("minor_correction")
                    last_feedback = "minor_correction"

                elif last_feedback != "major_correction":
                    feedback_placeholder.error("❌ Major correction needed.")
                    coach("major_correction")
                    last_feedback = "major_correction"

        else:
            feedback_placeholder.warning("⚠️ No pose detected.")
            if last_feedback != "pose_not_visible":
                coach("pose_not_visible")
                last_feedback = "pose_not_visible"

        cv2.imwrite(temp_file.name, frame)
        stframe.image(temp_file.name, channels="BGR", use_container_width=True)

        time.sleep(0.1)

    duration = round(time.time() - st.session_state.start_time, 2)

    log_session(
        pose=pose_name,
        reps=st.session_state.reps,
        feedback_list=list(st.session_state.feedback_collected),
        duration=duration
    )

    st.session_state.running = False
    st.session_state.stop = False

    fb = list(st.session_state.feedback_collected)
    st.success(f"✅ Session saved! Duration: {duration} sec | Reps: {st.session_state.reps} | Feedbacks: {len(fb)} types.")


def category_is_yoga(pose_name):
    return pose_name.lower() in ["tadasana", "vrikshasana"]
