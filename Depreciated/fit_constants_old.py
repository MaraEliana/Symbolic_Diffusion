import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import numpy as np
from scipy.optimize import minimize as scipy_minimize
# Assuming the input tensor might be a PyTorch tensor
# import torch # Uncomment if you need to handle torch tensor input explicitly

# Define default epsilon
DEFAULT_EPSILON = 1e-8

def create_rpn_fitting_function(rpn_tensor, vocabulary, xy_data, epsilon=DEFAULT_EPSILON):
    """
    Fits unknown constants 'C' in an RPN expression and returns a prediction function.

    Args:
        rpn_tensor: A tensor (e.g., PyTorch) containing integer tokens of the RPN expression.
        vocabulary: Dictionary mapping string tokens to integer indices.
        xy_data: NumPy array of shape (N, 3) where each row is [x1, x2, y_true].
        epsilon: Small value for safe logarithm log(abs(x) + epsilon).

    Returns:
        A callable function `predict(X)` where X is an array (M, 2) of input
        coordinates [x1, x2], and which returns a JAX array (M,) of predicted y values.
        Returns None if the RPN expression is invalid or optimization fails severely.
    """
    # --- 1. Preprocessing and Parsing ---
    try:
        # Ensure tensor is on CPU and convert to numpy for easier handling
        if hasattr(rpn_tensor, 'cpu'): # Check if it's a PyTorch tensor
             tokens = rpn_tensor.cpu().numpy()
        else: # Assume it's already numpy-like or a list
             tokens = np.asarray(rpn_tensor)

        # Ensure xy_data is numpy array
        xy_data = np.asarray(xy_data)
        if xy_data.ndim != 2 or xy_data.shape[1] != 3:
            raise ValueError("xy_data must be a 2D array with shape (N, 3)")

        x_data = jnp.array(xy_data[:, :2], dtype=jnp.float32)
        y_data = jnp.array(xy_data[:, 2], dtype=jnp.float32)

        inv_vocab = {v: k for k, v in vocabulary.items()}

        # Extract the active RPN tokens and count constants
        active_tokens = []
        num_constants = 0
        constant_token_idx = vocabulary.get("C")
        eos_token_idx = vocabulary.get("<EOS>")
        pad_token_idx = vocabulary.get("<PAD>")
        sos_token_idx = vocabulary.get("<SOS>")

        if constant_token_idx is None:
             raise ValueError("'C' token not found in vocabulary.")
        if eos_token_idx is None:
             raise ValueError("'<EOS>' token not found in vocabulary.")
        # PAD and SOS are optional for stopping but good practice

        for token_idx in tokens:
            if token_idx == sos_token_idx:
                continue
            if token_idx == eos_token_idx or token_idx == pad_token_idx:
                break # Stop processing at EOS or PAD

            token_str = inv_vocab.get(token_idx)
            if token_str is None:
                 print(f"Warning: Token index {token_idx} not found in vocabulary. Skipping.")
                 continue
            if token_str == "<UNK>":
                 raise ValueError("Encountered <UNK> token. Expression cannot be reliably evaluated.")

            active_tokens.append(token_idx)
            if token_idx == constant_token_idx:
                num_constants += 1

        if not active_tokens:
             raise ValueError("No valid tokens found in the RPN tensor after removing SOS/EOS/PAD.")

        print(f"Parsed RPN (indices): {active_tokens}")
        print(f"Number of constants 'C' to fit: {num_constants}")
        # Example RPN translation for clarity (optional)
        try:
            readable_rpn = [inv_vocab.get(t, f"UNK({t})") for t in active_tokens]
            print(f"Parsed RPN (tokens): {' '.join(readable_rpn)}")
        except Exception:
            print("Could not create readable RPN string.")


    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

    # --- 2. Define JAX RPN Evaluation Function ---
    # This function evaluates the expression for a single [x1, x2] point
    # and a given set of parameters (constants)
    def evaluate_rpn_single(x_point, params, current_active_tokens, current_inv_vocab, current_epsilon):
        stack = []
        const_param_idx = 0 # Index into the *params* array

        # Ensure x_point is usable
        x1 = x_point[0]
        x2 = x_point[1]

        for token_idx in current_active_tokens:
            token_str = current_inv_vocab.get(token_idx)
            # We assume valid tokens based on preprocessing

            try:
                if token_str == "C":
                    # Ensure params is indexable and has the correct size during tracing
                    if const_param_idx >= len(params):
                         # This error might occur if num_constants was miscalculated or params shape is wrong
                         # Returning NaN is often better for optimization than raising an error here.
                         print(f"Error: Trying to access param index {const_param_idx} but params length is {len(params)}")
                         return jnp.nan
                    stack.append(params[const_param_idx])
                    const_param_idx += 1
                elif token_str == "x1":
                    stack.append(x1)
                elif token_str == "x2":
                    stack.append(x2)
                # Handle literal numbers if they exist in vocab (treat as float)
                elif token_str.isdigit() or (token_str.startswith('-') and token_str[1:].isdigit()):
                     stack.append(jnp.array(float(token_str)))
                elif token_str == "2": # Specific handling if needed, prefer generic above
                     stack.append(jnp.array(2.0))
                elif token_str == "3":
                     stack.append(jnp.array(3.0))
                # Handle operators (ensure enough operands on stack)
                elif token_str in ["+", "-", "*", "/", "**"]:
                    if len(stack) < 2: return jnp.nan # Invalid RPN
                    op2 = stack.pop()
                    op1 = stack.pop()
                    if token_str == "+": stack.append(op1 + op2)
                    elif token_str == "-": stack.append(op1 - op2)
                    elif token_str == "*": stack.append(op1 * op2)
                    elif token_str == "/":
                        # Safe division (return NaN or large number for division by zero)
                        stack.append(jnp.where(op2 == 0, jnp.nan, op1 / op2))
                    elif token_str == "**":
                        # Power needs care (e.g., negative base to fractional power -> complex)
                        # JAX handles complex numbers if needed, but result might be complex.
                        # Let's assume real results are expected or handle potential NaNs later.
                        stack.append(op1 ** op2)
                # Handle unary functions
                elif token_str in ["sin", "cos", "exp", "log"]:
                    if len(stack) < 1: return jnp.nan # Invalid RPN
                    op1 = stack.pop()
                    op1_arr = jnp.asarray(op1) # Ensure it's a JAX array
                    if token_str == "sin": stack.append(jnp.sin(op1_arr))
                    elif token_str == "cos": stack.append(jnp.cos(op1_arr))
                    elif token_str == "exp": stack.append(jnp.exp(op1_arr))
                    elif token_str == "log":
                        # Safe log: log(abs(x) + eps)
                        stack.append(jnp.log(jnp.abs(op1_arr) + current_epsilon))
                else:
                    # Should not happen if vocab is correct and UNK handled
                    print(f"Warning: Unhandled token '{token_str}' during evaluation.")
                    return jnp.nan

            except IndexError: # Catch stack underflow
                 print(f"Error: Stack underflow during evaluation for token '{token_str}'. Invalid RPN?")
                 return jnp.nan
            except Exception as e: # Catch other potential errors during evaluation
                 print(f"Error during RPN evaluation step with token '{token_str}': {e}")
                 return jnp.nan


        # Final check: stack should contain exactly one result
        if len(stack) != 1:
            # print(f"Warning: Final stack size is {len(stack)}, expected 1. Stack: {stack}")
            return jnp.nan # Indicate evaluation failure

        result = jnp.asarray(stack[0]) # Ensure result is a JAX array

        # Optional: Check for NaN/Inf in the final result itself
        # result = jnp.where(jnp.isnan(result) | jnp.isinf(result), jnp.nan, result)

        return result

    # --- 3. Vectorize the Evaluation Function ---
    # Vectorize over the first argument (x_point), keep others constant
    evaluate_rpn_batch = jax.vmap(
        evaluate_rpn_single,
        in_axes=(0, None, None, None, None) # Map over x_data, keep params and others fixed per call
    )

    # --- 4. Define the Loss Function ---
    # @jax.jit # Optional: JIT-compile the loss function for speed
    def loss_fn(params, x_batch, y_batch, loss_active_tokens, loss_inv_vocab, loss_epsilon):
        y_pred = evaluate_rpn_batch(x_batch, params, loss_active_tokens, loss_inv_vocab, loss_epsilon)

        # Handle potential NaNs/Infs from evaluation before calculating loss
        # Assign a large penalty for invalid evaluations (NaN/Inf)
        # This encourages the optimizer to find constants that lead to valid results.
        is_invalid = jnp.isnan(y_pred) | jnp.isinf(y_pred)
        squared_error = jnp.where(is_invalid, 1e6, (y_pred - y_batch)**2) # Large penalty

        return jnp.mean(squared_error)

    # --- 5. Perform Optimization ---
    if num_constants == 0:
        print("No constants 'C' found. Returning prediction function with no fitted parameters.")
        # Check if the expression is valid without constants
        try:
             test_pred = evaluate_rpn_batch(x_data[:1], jnp.array([]), active_tokens, inv_vocab, epsilon)
             if jnp.isnan(test_pred).any() or jnp.isinf(test_pred).any():
                   print("Warning: Expression evaluation results in NaN/Inf even without constants.")
             optimized_params = jnp.array([], dtype=jnp.float32)
        except Exception as e:
             print(f"Error evaluating expression even without constants: {e}")
             return None

    else:
        print("Optimizing constants...")
        initial_params = jnp.ones(num_constants, dtype=jnp.float32) # Initial guess

        # Define args tuple carefully matching loss_fn signature (excluding params)
        optimizer_args = (x_data, y_data, active_tokens, inv_vocab, epsilon)

        try:
            optimizer_result = scipy_minimize(
                loss_fn,
                initial_params,
                args=optimizer_args,
                method='L-BFGS-B',
                # options={'disp': True, 'maxiter': 1000} # Optional: display progress, set max iterations
                 options={'maxiter': 2000}
            )

            if not optimizer_result.success:
                print(f"Warning: Optimization did not converge! Message: {optimizer_result.message}")
                # Decide whether to proceed with the result or return None
                # Let's proceed but warn the user.

            optimized_params = jnp.array(optimizer_result.x, dtype=jnp.float32)
            final_loss = optimizer_result.fun

            print(f"Optimization finished.")
            print(f"  Success: {optimizer_result.success}")
            print(f"  Status: {optimizer_result.status}")
            print(f"  Message: {optimizer_result.message}")
            print(f"  Final Loss: {final_loss:.6f}")
            print(f"  Fitted Constants: {optimized_params}")

            # Optional: Check if final loss is reasonable (e.g., not NaN or excessively large)
            if jnp.isnan(final_loss) or jnp.isinf(final_loss) or final_loss > 1e5:
                 print("Warning: Final loss is very high or invalid. The fit might be poor.")


        except Exception as e:
            print(f"Error during optimization: {e}")
            # Attempting to evaluate the function with initial parameters might give more info
            try:
                initial_loss = loss_fn(initial_params, *optimizer_args)
                print(f"Loss with initial params: {initial_loss}")
                if jnp.isnan(initial_loss):
                     print("Loss is NaN even with initial parameters. Check RPN evaluation logic and input data.")
            except Exception as eval_e:
                print(f"Error evaluating loss function even with initial params: {eval_e}")
            return None # Optimization failed

    # --- 6. Create and Return the Prediction Function ---
    # This function closes over the optimized parameters and other necessary info
    def prediction_function(new_x_coords):
        """
        Predicts y values for new x coordinates using the fitted RPN expression.

        Args:
            new_x_coords: A NumPy or JAX array of shape (M, 2) or a list/tuple
                          representing a single coordinate pair [x1, x2].

        Returns:
            A JAX array of predicted y values, shape (M,) or a scalar if input is single point.
        """
        try:
            # Ensure input is a JAX array with correct dtype and shape
            new_x_coords = jnp.asarray(new_x_coords, dtype=jnp.float32)

            if new_x_coords.ndim == 1:
                # Assume it's a single [x1, x2] pair
                if new_x_coords.shape != (2,):
                    raise ValueError("Single input must be of shape (2,)")
                # Reshape to (1, 2) for batch evaluation
                new_x_coords_batch = new_x_coords.reshape(1, -1)
                # Use the vectorized evaluator
                y_pred_batch = evaluate_rpn_batch(
                    new_x_coords_batch, optimized_params, active_tokens, inv_vocab, epsilon
                )
                # Return the single scalar result
                return y_pred_batch[0]
            elif new_x_coords.ndim == 2:
                # Assume it's a batch of coordinates
                if new_x_coords.shape[1] != 2:
                    raise ValueError("Batch input must be of shape (M, 2)")
                # Use the vectorized evaluator directly
                y_pred_batch = evaluate_rpn_batch(
                    new_x_coords, optimized_params, active_tokens, inv_vocab, epsilon
                )
                return y_pred_batch
            else:
                raise ValueError("Input must be a single coordinate (1D array/list of length 2) or a batch (2D array Nx2)")

        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return NaN or raise? Returning NaN might be safer for some pipelines.
            # Determine expected output shape based on input to return NaNs correctly.
            if new_x_coords.ndim == 1:
                return jnp.nan
            elif new_x_coords.ndim == 2:
                return jnp.full(new_x_coords.shape[0], jnp.nan)
            else: # Should have been caught earlier, but as a fallback
                raise e # Re-raise if shape was truly unexpected

    return prediction_function