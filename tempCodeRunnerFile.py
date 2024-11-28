if predicted_label == 'unsafe':
    #     unsafe_class_index = label_encoder.transform(['unsafe'])[0]
    #     unsafe_probability = prediction[0][unsafe_class_index]
    #     threat_level = 'High' if unsafe_probability > 0.8 else 'Medium' if unsafe_probability > 0.5 else 'Low'
    #     print(f"Threat Level: {threat_level} (Confidence: {unsafe_probability:.2f})")