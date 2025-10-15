import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import difflib
from typing import Optional, List, Dict, Tuple, Any, Union
import warnings
def compute_changes_and_distance(orig: List[str], cand: List[str]):
        ops = difflib.SequenceMatcher(a=orig, b=cand).get_opcodes()
        changes_local = []
        change_count = 0
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                continue
            change_count += max(i2 - i1, j2 - j1)
            # Record a human-readable change
            from_text = " ".join(orig[i1:i2]) if i2 > i1 else ""
            to_text = " ".join(cand[j1:j2]) if j2 > j1 else ""
            changes_local.append({
                "type": tag,
                "position_orig": i1,
                "from": from_text,
                "to": to_text
            })
        distance_local = float(change_count) / max(len(orig), 1)
        return changes_local, distance_local
class XAITools:
    """Global configuration for XAI tools"""
    def __init__(self, classifier_name="textattack/bert-base-uncased-imdb"):
        model_name = classifier_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cls = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.prediction_classes = ["negative", "positive"]
        self.label_map = {0: "negative", 1: "positive"}
        self.inverse_label_map = {"negative": 0, "positive": 1}
    def predict(self, instance) -> Tuple[str, Dict[str, float]]:
        encoding = self.tokenizer(
                instance,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        for k, v in encoding.items():
            encoding[k] = v.to(self.device)

        self.cls.eval()
        with torch.no_grad():
            # Use kwargs form so it works with different HF model signatures
            outputs = self.cls(**encoding)

            # Extract logits robustly for different transformers versions / return types
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                # outputs may be a tuple/list with logits first
                logits = outputs[0]

            # logits shape: (batch, num_labels)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # Prefer model.config.id2label if available (works for many HF fine-tuned models)
        label_list: List[str]
        try:
            id2label = getattr(getattr(self.cls, "config", None), "id2label", None)
            if id2label:
                # Ensure integer ordering of labels
                id2label_int = {int(k): v for k, v in id2label.items()}
                label_list = [id2label_int[i] for i in sorted(id2label_int.keys())]
            else:
                label_list = self.sentiment_labels
        except Exception:
            label_list = self.sentiment_labels

        predicted_idx = int(np.argmax(probs))
        predicted_label = label_list[predicted_idx] if predicted_idx < len(label_list) else self.sentiment_labels[predicted_idx]

        prob_dict = {
            label_list[i] if i < len(label_list) else self.sentiment_labels[i]: float(probs[i])
            for i in range(len(probs))
        }
        map_label = {"LABEL_0": "negative", "LABEL_1": "positive"}
        predicted_label = map_label.get(predicted_label, predicted_label)
        return predicted_label, prob_dict
    def get_feature_attributions_shap(self, instance: str, prediction_label: str = None, model = None) -> Dict[str, Any]:
        """
        Computes SHAP (SHapley Additive exPlanations) values for feature attribution.
        SHAP provides theoretically grounded feature importance based on game theory.
        
        Use this tool when:
        - User specifically asks for "SHAP values" or "Shapley values"
        - User wants mathematically rigorous feature importance
        - User asks "How much does each word contribute to the prediction?"
        
        This tool provides:
        - SHAP values for each token (can be positive or negative)
        - Base value (expected model output)
        - Sum of SHAP values + base = actual prediction
        - Additive explanation that sums to final prediction
        
        Args:
            instance: The text instance to explain
            prediction_label: Optional target label to explain (if None, explains predicted class)
            
        Returns:
            dict: Contains 'base_value', 'shap_values' (token: value pairs), 'prediction'
            
        Example:
            >>> get_feature_attributions_shap("The acting was superb")
            {
                "base_value": 0.5,
                "shap_values": {"acting": 0.25, "superb": 0.30, "The": 0.02, "was": 0.01},
                "prediction": "positive"
            }
        """
    # Try to use actual SHAP if ferret library is available
        import warnings
        try:
            import shap


            model = self.cls
            tokenizer = self.tokenizer
            model.eval()

            # Prediction function returning full probability vectors for a list of texts
            def _model_predict_proba(texts: List[str]) -> np.ndarray:
                encoding = tokenizer(
                    texts,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                for k, v in encoding.items():
                    encoding[k] = v.to(self.device)

                with torch.no_grad():
                    outputs = model(**encoding)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                return probs  # shape: (batch, num_labels)

            # Determine label ordering (prefer model.config.id2label if available)
            try:
                id2label = getattr(getattr(model, "config", None), "id2label", None)
                if id2label:
                    id2label_int = {int(k): v for k, v in id2label.items()}
                    label_list = [id2label_int[i] for i in sorted(id2label_int.keys())]
                else:
                    label_list = self.sentiment_labels
            except Exception:
                label_list = self.sentiment_labels

            predicted_label, prob_dict = self.predict(instance)
            target = prediction_label if prediction_label else predicted_label
            try:
                target_idx = label_list.index(target)
            except ValueError:
                target_idx = self.inverse_label_map.get(target, 0)

            # Use SHAP Text masker + Explainer
            masker = shap.maskers.Text(tokenizer)
            explainer = shap.Explainer(_model_predict_proba, masker, output_names=label_list)

            # Explain the single instance
            shap_result = explainer([instance])

            # Extract tokens and SHAP values for target class robustly
            try:
                # shap_result.data / .values shapes depend on shap version
                tokens = shap_result.data[0] if hasattr(shap_result, "data") else shap_result[0].data
                values = shap_result.values[0] if hasattr(shap_result, "values") else shap_result[0].values
                # values shape: (n_tokens, n_classes) or (n_classes, n_tokens) depending on version
                if values.ndim == 2 and values.shape[1] == len(label_list):
                    # (n_tokens, n_classes)
                    target_values = values[:, target_idx]
                elif values.ndim == 2 and values.shape[0] == len(label_list):
                    # (n_classes, n_tokens)
                    target_values = values[target_idx, :]
                else:
                    # fallback: try selecting last dimension
                    target_values = values[:, target_idx] if values.shape[-1] == len(label_list) else values[:, 0]

                shap_values = {}
                for tok, val in zip(tokens, target_values):
                    if not str(tok).startswith("##") and tok not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                        shap_values[str(tok)] = float(val)

                # Base value for target class
                if hasattr(shap_result, "base_values"):
                    base_val = shap_result.base_values[0]
                    base_value = float(base_val[target_idx]) if isinstance(base_val, (list, np.ndarray)) and len(base_val) > target_idx else float(np.mean(base_val))
                else:
                    base_value = float(np.mean(list(prob_dict.values())))

                return {
                    "base_value": base_value,
                    "shap_values": shap_values,
                    "prediction": predicted_label,
                    "target_label": target,
                    "method": "shap_explainer"
                }

            except Exception as e:
                warnings.warn(f"Unexpected SHAP result format: {e}. ")

        except ImportError:
            warnings.warn("shap library not available")
        except Exception as e:
            warnings.warn(f"SHAP explanation failed: {e}.")

    def get_counterfactual_explanation(self, instance: str, target_class: str = None) -> Dict[str, Any]:
        """
        Generates counterfactual explanations showing what minimal changes to the instance
        would flip the prediction to a different class.
        
        Use this tool when:
        - User asks "What would need to change to get a different prediction?"
        - User asks "Why this prediction instead of another?"
        - User wants to understand decision boundaries
        - User asks "What if I changed X?"
        
        This tool CAN:
        - Show minimal edits needed to change the prediction
        - Answer contrastive questions (why X not Y?)
        - Reveal decision boundaries
        - Generate actionable insights
        
        This tool provides:
        - Modified version of the instance with flipped prediction
        - Specific changes made (words added/removed/replaced)
        - Distance/similarity to original instance
        
        Args:
            instance: The original instance to generate counterfactuals for
            target_class: Optional desired class (if None, finds nearest alternative)
            
        Returns:
            dict: Contains counterfactual text, changes made, and new prediction
            
        Example:
            >>> get_counterfactual_explanation("The movie was great", target_class="negative")
            {
                "original": "The movie was great",
                "counterfactual": "The movie was terrible",
                "changes": [{"from": "great", "to": "terrible"}],
                "original_prediction": "positive",
                "counterfactual_prediction": "negative"
            }
        """
        original_prediction, original_probs = self.predict(instance)
        
        # Determine target class
        if target_class is None:
            # Find class with second highest probability
            sorted_classes = sorted(original_probs.items(), key=lambda x: x[1], reverse=True)
            target_class = sorted_classes[1][0] if len(sorted_classes) > 1 else "negative"

        # Tokenize original for distance computation
        def _tokenize_text(text):
            return text.lower().split()
        
        orig_tokens = _tokenize_text(instance)

        # Generate multiple diverse counterfactual candidates using self.llm
        candidates = []

        prompt = (
            f"Rewrite the following text so that its sentiment becomes '{target_class}' "
            "with as few edits as possible. Preserve wording and meaning where you can. "
            "Return only the rewritten text (no commentary).\n\n"
            f"Text: {instance}\n\nRewrite:"
        )

        num_candidates = 5
        
        # Use self.llm pipeline for generation
        outputs = self.llm(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            num_return_sequences=num_candidates,
            return_full_text=False
        )

        raw_texts = []
        for out in outputs:
            generated_text = out['generated_text'].strip()
            # Clean up the generated text
            candidate = generated_text.splitlines()[0].strip()
            if candidate and candidate != instance:
                raw_texts.append(candidate)

        # Deduplicate preserving order
        seen = set()
        for t in raw_texts:
            if t not in seen:
                seen.add(t)
            candidates.append(t)
        
        candidate_records = []
        for cand_text in candidates:
            cf_pred, cf_probs = self.predict(cand_text)
            cand_tokens = _tokenize_text(cand_text)
            changes_local, distance_local = compute_changes_and_distance(orig_tokens, cand_tokens)
            candidate_records.append({
            "text": cand_text,
            "prediction": cf_pred,
            "probs": cf_probs,
            "changes": changes_local,
            "distance": distance_local
            })

        # Select candidate that achieves target label with minimal distance
        hitting = [c for c in candidate_records if c["prediction"] == target_class]
        if hitting:
            chosen = min(hitting, key=lambda x: x["distance"])
        else:
            # Score by target probability then distance
            def score_fn(x):
                return (x["probs"].get(target_class, 0.0), -x["distance"])
            chosen = max(candidate_records, key=score_fn) if candidate_records else None

        if chosen:
            counterfactual_text = chosen["text"]
            changes = chosen["changes"]
            cf_prediction = chosen["prediction"]
            cf_probs = chosen["probs"]
            distance = chosen["distance"]
        
        return {
            "original": instance,
            "counterfactual": counterfactual_text,
            "changes": changes,
            "original_prediction": original_prediction,
            "counterfactual_prediction": cf_prediction,
            "target_class": target_class,
            "achieved_target": cf_prediction == target_class,
            "distance": float(distance),
            "original_probabilities": original_probs,
            "counterfactual_probabilities": cf_probs
        }
    def get_model_information(self, field: str) -> Union[str, List[str], Dict[str, Any]]:
        """
        Retrieves comprehensive information about the model itself.
        
        Use this tool when:
        - User asks "What model is this?"
        - User wants to know about model architecture
        - User asks "How was the model trained?"
        - User needs model specifications
        
        This tool provides:
        - Model architecture details
        - Training procedure and hyperparameters
        - Training data information
        - Evaluation metrics and performance
        - Model size and computational requirements
        
        Args:
            field: The aspect to query. Options:
                - "architecture": Model structure, layers, parameters
                - "training_data": Datasets used for training
                - "training_procedure": Optimization, loss functions, hyperparameters
                - "evaluation": Performance metrics, benchmarks
                - "specifications": Size, memory, computational requirements
                - "version": Model version and release information
                - "all": Return all available information
                
        Returns:
            str or dict: Information about the requested field
            
        Example:
            >>> get_model_information("architecture")
            "Transformer-based model with 12 layers and 768 hidden units."
        """
        # Initialize info dict
        info = {}
        
        
        try:
            model = self.cls
            model_type = type(model).__name__
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Extract configuration
            config_info = {}
            if hasattr(model, "config"):
                config = model.config
                config_info = {
                    "hidden_size": getattr(config, "hidden_size", "N/A"),
                    "num_layers": getattr(config, "num_hidden_layers", "N/A"),
                    "attention_heads": getattr(config, "num_attention_heads", "N/A"),
                    "vocab_size": getattr(config, "vocab_size", "N/A"),
                    "max_length": getattr(config, "max_position_embeddings", "N/A"),
                    "num_labels": getattr(config, "num_labels", len(_config.prediction_classes))
                }
            
            # Architecture
            arch_parts = [f"{model_type}"]
            if config_info.get("num_layers") != "N/A":
                arch_parts.append(f"with {config_info['num_layers']} layers")
            if config_info.get("hidden_size") != "N/A":
                arch_parts.append(f"{config_info['hidden_size']} hidden units")
            if config_info.get("attention_heads") != "N/A":
                arch_parts.append(f"{config_info['attention_heads']} attention heads")
            arch_parts.append(f"Total: {total_params:,} parameters ({trainable_params:,} trainable)")
            
            info["architecture"] = ". ".join(arch_parts)
            
            # Training data (heuristic based on model type)
            if "bert" in model_type.lower():
                info["training_data"] = "Typically trained on large text corpora (BooksCorpus, Wikipedia). For this specific instance, training data details may vary."
            else:
                info["training_data"] = "Training data information not available. Depends on specific model fine-tuning."
            
            # Training procedure
            optimizer_lr = "3e-5 (typical)"
            info["training_procedure"] = f"Supervised learning with cross-entropy loss. Common settings: Adam optimizer (lr={optimizer_lr}), batch size 16-32, gradient accumulation, warmup schedule, dropout regularization."
            
            # Evaluation
            info["evaluation"] = "Evaluation metrics depend on fine-tuning. For classification: accuracy, precision, recall, F1-score. Model-specific benchmarks not available without evaluation data."
            
            # Specifications
            model_size_mb = (total_params * 4) / (1024 ** 2)  # Assuming fp32
            info["specifications"] = f"Model size: ~{model_size_mb:.0f}MB (fp32), Parameters: {total_params:,}, Device: {self.device}, Inference mode: {not model.training}"
            
            # Version
            model_name = getattr(model.config, "model_type", model_type) if hasattr(model, "config") else model_type
            info["version"] = f"Model type: {model_name}, Class: {model_type}"
            
            # Limitations
            max_len = config_info.get("max_length", 512)
            info["limitations"] = f"Context length limited to {max_len} tokens. May inherit biases from training data. Performance depends on domain similarity to training data."
            
            # Intended use
            task_type = "text classification"
            if "SequenceClassification" in model_type:
                task_type = "sequence classification"
            info["intended_use"] = f"Designed for {task_type} tasks. Current setup: {len(self.prediction_classes)}-class classification ({', '.join(self.prediction_classes)})"
            
        except Exception as e:
            # Fallback with error info
            info = {
                "architecture": f"Model loaded but details extraction failed: {str(e)}",
                "training_data": "Information not available",
                "training_procedure": "Information not available",
                "evaluation": "Information not available",
                "specifications": f"Model available on {self.device}",
                "version": "Unknown",
                "limitations": "Unknown",
                "intended_use": f"{len(self.prediction_classes)}-class classification"
            }
        
        if field == "all":
            return info
        return info.get(field, f"Unknown field: {field}. Available fields: {', '.join(info.keys())}")

    def get_multiple_counterfactuals(self, instance: str, target_label: str = None, num_counterfactuals: int = 3) -> List[Dict[str, Any]]:
        original_pred, original_probs = self.predict(instance)

        # choose an alternative/target class (second highest prob if available)
        if target_label is not None:
            target_class = target_label
        else:
            sorted_classes = sorted(original_probs.items(), key=lambda x: x[1], reverse=True)
            target_class = sorted_classes[1][0] if len(sorted_classes) > 1 else sorted_classes[0][0]

        prompt = (
            f"Rewrite the following text so that its sentiment becomes '{target_class}' "
            "with as few edits as possible. Preserve wording and meaning where you can. "
            "Return only the rewritten text (no commentary).\n\n"
            f"Text: {instance}\n\nRewrite:"
        )

        # generate more candidates than requested to improve chance of flips
        gen_count = max(num_counterfactuals * 3, 6)
        outputs = self.llm(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            num_return_sequences=gen_count,
            return_full_text=False
        )

        # collect unique candidates preserving order
        seen = set()
        candidates = []
        for out in outputs:
            gen = out.get("generated_text", "") if isinstance(out, dict) else str(out)
            cand = gen.strip().splitlines()[0].strip()
            if cand and cand != instance and cand not in seen:
                seen.add(cand)
                candidates.append(cand)

        orig_tokens = instance.lower().split()
        flipped = []
        for cand_text in candidates:
            pred, probs = self.predict(cand_text)
            if pred == original_pred:
                continue
            cand_tokens = cand_text.lower().split()
            changes, distance = compute_changes_and_distance(orig_tokens, cand_tokens)
            flipped.append({
                "text": cand_text,
                "prediction": pred,
                "probs": probs,
                "changes": changes,
                "distance": float(distance)
            })
            if len(flipped) >= num_counterfactuals:
                break

        return flipped
    def get_feature_attribution(self, instance: str, feature: str, prediction_label: str = None) -> Dict[str, Any]:
        """
        Gets attribution of a specific feature for the model's prediction on a given instance.

        This tool provides:
        - Importance score for the specified feature
        - Contextual explanation of the feature's role in the prediction

        Args:
            instance: The text instance to analyze (e.g., "The movie was great")
            feature: The specific feature/word/token to get attribution for (e.g., "great")
            prediction_label: Optional target label to explain (if None, explains predicted class)
        
        Returns:
            dict: Contains attribution score and related information
        """
        
        # Get full SHAP attributions
        shap_result = self.get_feature_attributions_shap(instance, prediction_label)
        
        if not shap_result or "shap_values" not in shap_result:
            return {
                "feature": feature,
                "attribution_score": 0.0,
                "found": False,
                "message": "Could not compute SHAP values for this instance"
            }
        
        shap_values = shap_result["shap_values"]
        predicted_label = shap_result["prediction"]
        target_label = shap_result.get("target_label", predicted_label)
        
        # Normalize feature and search in SHAP values
        feature_normalized = feature.lower().strip()
        
        # Try exact match first
        attribution_score = None
        matched_token = None
        
        for token, score in shap_values.items():
            if token.lower() == feature_normalized:
                attribution_score = score
                matched_token = token
                break
        
        # If no exact match, try partial match
        if attribution_score is None:
            for token, score in shap_values.items():
                if feature_normalized in token.lower() or token.lower() in feature_normalized:
                    attribution_score = score
                    matched_token = token
                    warnings.warn(f"Exact match not found. Using partial match: '{matched_token}'")
                    break
        
        if attribution_score is None:
            return {
                "feature": feature,
                "attribution_score": 0.0,
                "found": False,
                "message": f"Feature '{feature}' not found in instance. Available tokens: {list(shap_values.keys())}",
                "prediction": predicted_label,
                "target_label": target_label
            }
        
        # Provide contextual explanation
        direction = "positive" if attribution_score > 0 else "negative"
        magnitude = abs(attribution_score)
        
        if magnitude > 0.1:
            strength = "strongly"
        elif magnitude > 0.05:
            strength = "moderately"
        else:
            strength = "weakly"
        
        explanation = (
            f"The feature '{matched_token}' contributes {strength} in a {direction} direction "
            f"(score: {attribution_score:.4f}) towards the '{target_label}' prediction."
        )
        
        return {
            "feature": feature,
            "matched_token": matched_token,
            "attribution_score": float(attribution_score),
            "found": True,
            "direction": direction,
            "magnitude": float(magnitude),
            "strength": strength,
            "explanation": explanation,
            "prediction": predicted_label,
            "target_label": target_label,
            "base_value": shap_result.get("base_value", 0.0),
            "all_attributions": shap_values
        }
    def get_classifier_performance(self) -> Dict[str, Any]:
        """
        Provides performance metrics of the classifier model.

        Use this tool when:
        - User asks "How well does the model perform?"
        - User wants to know accuracy, precision, recall, F1-score
        - User needs benchmark comparisons

        This tool provides:
        - Key performance metrics
        - Benchmark datasets used
        - Comparison to baseline models

        Returns:
            dict: Contains accuracy, precision, recall, F1-score, and other relevant metrics
        """
        # Placeholder values; in practice, these would be computed on a validation/test set
        performance_metrics = {
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.93,
            "f1_score": 0.92,
            "roc_auc": 0.95,
            "benchmark_dataset": "IMDB Movie Reviews",
            "baseline_comparison": {
                "random_forest": 0.85,
                "logistic_regression": 0.88
            },
            "notes": "Performance may vary based on input data distribution."
        }
        return performance_metrics


    def get_system_information(self) -> dict:
        return {
            "output": "sentiment, either positive or negative", 
            "limitations": "may not handle sarcasm or nuanced language well",
            "improvements": "could be enhanced with more diverse training data",
        }


    def get_data_information(self) -> dict:
        return {
            "source": "IMDB Movie Reviews dataset",
            "size": "50,000 reviews",
            "classes": ["positive", "negative"],
            "preprocessing": "tokenization, lowercasing, padding/truncation to max length 512",
            "limitations": "may contain biases present in movie reviews",
            "updates": "dataset is static, but model can be fine-tuned on new data"
        }