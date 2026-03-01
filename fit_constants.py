import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution # Using DE

# Define default epsilon
DEFAULT_EPSILON = 1e-8

def create_rpn_fitting_function(rpn_tensor,
                                vocabulary,
                                xy_data,
                                epsilon=DEFAULT_EPSILON,
                                constant_bounds=(-10.0, 10.0),
                                de_maxiter=100,          # Max generations for DE (can be tuned for speed vs quality)
                                de_popsize_factor=15,    # Popsize factor (can be tuned)
                                de_strategy='best1bin',  # DE strategy
                                de_mutation=(0.5, 1.0),  # DE mutation range
                                de_recombination=0.7,    # DE recombination rate
                                de_tol=0.01,             # Relative tolerance for convergence
                                de_polish=True,         # Set False to avoid L-BFGS-B post-processing if DE is preferred standalone
                                de_workers=1,            # Number of parallel workers for DE (-1 for all cores)
                                de_disp=False):          # Display DE progress
    """
    Fits unknown constants 'C' in an RPN expression using Differential Evolution
    and returns a prediction function.

    Args:
        rpn_tensor: A tensor (e.g., PyTorch) containing integer tokens of the RPN expression.
        vocabulary: Dictionary mapping string tokens to integer indices.
        xy_data: NumPy array of shape (N, 3) where each row is [x1, x2, y_true].
        epsilon: Small value for safe logarithm log(abs(x) + epsilon).
        constant_bounds: Tuple (min, max) for the bounds of each constant 'C'.
        de_maxiter: Max generations for DE. Lower for speed, higher for quality.
        de_popsize_factor: DE population size = factor * num_constants. Lower for speed.
        de_strategy: DE strategy (e.g., 'best1bin', 'rand1bin').
        de_mutation: DE mutation factor or tuple (min, max).
        de_recombination: DE recombination probability.
        de_tol: Relative tolerance for convergence. Higher for faster, rougher convergence.
        de_polish: If True, L-BFGS-B is used to polish the best DE solution.
                   Set to False if you want to rely purely on DE or if L-BFGS is too slow.
        de_workers: Number of cores for DE to use (-1 for all available).
                    Set > 1 for potential speedup on multi-core systems.
        de_disp: If True, print convergence messages from DE.

    Returns:
        A callable function `predict(X)` where X is an array (M, 2) of input
        coordinates [x1, x2], and which returns a JAX array (M,) of predicted y values.
        Returns None if the RPN expression is invalid or optimization fails severely.
    """
    # --- 1. Preprocessing and Parsing ---
    # These variables will be "baked into" the JITted loss function via closure
    _x_data_for_loss, _y_data_for_loss = None, None
    _active_tokens_for_loss, _inv_vocab_for_loss, _epsilon_for_loss = None, None, None
    num_constants = 0

    try:
        if hasattr(rpn_tensor, 'cpu'):
             tokens_np = rpn_tensor.cpu().numpy()
        else:
             tokens_np = np.asarray(rpn_tensor)

        xy_data_np = np.asarray(xy_data)
        if xy_data_np.ndim != 2 or xy_data_np.shape[1] != 3:
            raise ValueError("xy_data must be a 2D array with shape (N, 3)")

        _x_data_for_loss = jnp.array(xy_data_np[:, :2], dtype=jnp.float32)
        _y_data_for_loss = jnp.array(xy_data_np[:, 2], dtype=jnp.float32)
        _inv_vocab_for_loss = {v: k for k, v in vocabulary.items()}
        _epsilon_for_loss = float(epsilon) # Ensure it's a Python float for consistency

        _active_tokens_for_loss = []
        constant_token_idx = vocabulary.get("C")
        eos_token_idx = vocabulary.get("<EOS>")
        pad_token_idx = vocabulary.get("<PAD>")
        sos_token_idx = vocabulary.get("<SOS>")

        if constant_token_idx is None: raise ValueError("'C' token not found in vocabulary.")
        if eos_token_idx is None: raise ValueError("'<EOS>' token not found in vocabulary.")

        for token_idx_val in tokens_np:
            if token_idx_val == sos_token_idx: continue
            if token_idx_val == eos_token_idx or token_idx_val == pad_token_idx: break
            token_str_val = _inv_vocab_for_loss.get(token_idx_val)
            if token_str_val is None:
                 print(f"Warning: Token index {token_idx_val} not found in vocabulary. Skipping.")
                 continue
            if token_str_val == "<UNK>": raise ValueError("Encountered <UNK> token.")
            _active_tokens_for_loss.append(token_idx_val) # Appending int token
            if token_idx_val == constant_token_idx: num_constants += 1

        if not _active_tokens_for_loss: raise ValueError("No valid tokens found.")

        print(f"Parsed RPN (indices): {_active_tokens_for_loss}")
        print(f"Number of constants 'C' to fit: {num_constants}")
        try:
            readable_rpn = [_inv_vocab_for_loss.get(t, f"UNK({t})") for t in _active_tokens_for_loss]
            print(f"Parsed RPN (tokens): {' '.join(readable_rpn)}")
        except Exception: print("Could not create readable RPN string.")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

    # --- 2. Define JAX RPN Evaluation Function ---
    # This function is not JITted directly as a whole due to Python control flow based on strings.
    # However, jax.vmap will JIT-compile the JAX operations within it for batched execution.
    def evaluate_rpn_single(x_point, params, current_active_tokens_list, current_inv_vocab_dict, current_epsilon_float):
        stack = []
        const_param_idx = 0
        x1, x2 = x_point[0], x_point[1]

        for token_idx in current_active_tokens_list:
            token_str = current_inv_vocab_dict.get(token_idx) # Use passed dict
            try:
                if token_str == "C":
                    if const_param_idx >= len(params): return jnp.nan
                    stack.append(params[const_param_idx])
                    const_param_idx += 1
                elif token_str == "x1": stack.append(x1)
                elif token_str == "x2": stack.append(x2)
                elif token_str.isdigit() or (token_str.startswith('-') and token_str[1:].isdigit()):
                     stack.append(jnp.array(float(token_str)))
                elif token_str == "2": stack.append(jnp.array(2.0))
                elif token_str == "3": stack.append(jnp.array(3.0))
                elif token_str in ["+", "-", "*", "/", "**"]:
                    if len(stack) < 2: return jnp.nan
                    op2, op1 = stack.pop(), stack.pop()
                    if token_str == "+": stack.append(op1 + op2)
                    elif token_str == "-": stack.append(op1 - op2)
                    elif token_str == "*": stack.append(op1 * op2)
                    elif token_str == "/": stack.append(jnp.where(op2 == 0, jnp.nan, op1 / op2))
                    elif token_str == "**": stack.append(op1 ** op2)
                elif token_str in ["sin", "cos", "exp", "log"]:
                    if len(stack) < 1: return jnp.nan
                    op1_arr = jnp.asarray(stack.pop())
                    if token_str == "sin": stack.append(jnp.sin(op1_arr))
                    elif token_str == "cos": stack.append(jnp.cos(op1_arr))
                    elif token_str == "exp": stack.append(jnp.exp(op1_arr))
                    elif token_str == "log": stack.append(jnp.log(jnp.abs(op1_arr) + current_epsilon_float)) # Use passed epsilon
                else:
                    # print(f"Warning: Unhandled token '{token_str}' during evaluation.") # Can be noisy
                    return jnp.nan
            except IndexError: return jnp.nan
            except Exception: return jnp.nan # General eval error

        if len(stack) != 1: return jnp.nan
        result = jnp.asarray(stack[0])
        return result

    # --- 3. Vectorize the Evaluation Function ---
    evaluate_rpn_batch = jax.vmap(
        evaluate_rpn_single,
        in_axes=(0, None, None, None, None)
    )

    # --- 4. Define and JIT the Core Loss Calculation ---
    # This function closes over _x_data_for_loss, _y_data_for_loss,
    # _active_tokens_for_loss, _inv_vocab_for_loss, _epsilon_for_loss,
    # which are fixed for this particular RPN fitting task.
    @jax.jit
    def jitted_core_loss_calculation(params_jax):
        y_pred = evaluate_rpn_batch(_x_data_for_loss,
                                    params_jax,
                                    _active_tokens_for_loss,
                                    _inv_vocab_for_loss,
                                    _epsilon_for_loss)
        is_invalid = jnp.isnan(y_pred) | jnp.isinf(y_pred)
        penalty = jnp.array(1e6, dtype=_y_data_for_loss.dtype)
        squared_error = jnp.where(is_invalid, penalty, (y_pred - _y_data_for_loss)**2)
        mean_squared_error = jnp.mean(squared_error)
        return mean_squared_error

    # Wrapper function to be called by SciPy's optimizer
    def loss_fn_for_scipy(params_np):
        params_jax = jnp.asarray(params_np, dtype=jnp.float32) # DE works with float64 by default sometimes
        loss_value = jitted_core_loss_calculation(params_jax)
        return float(loss_value) # SciPy optimizers expect a standard Python float

    # --- 5. Perform Optimization ---
    optimized_params = None # Define in this scope
    if num_constants == 0:
        print("No constants 'C' found. Returning prediction function with no fitted parameters.")
        try:
             test_pred = evaluate_rpn_batch(_x_data_for_loss[:1], jnp.array([]),
                                            _active_tokens_for_loss, _inv_vocab_for_loss, _epsilon_for_loss)
             if jnp.isnan(test_pred).any() or jnp.isinf(test_pred).any():
                   print("Warning: Expression evaluation results in NaN/Inf even without constants.")
             optimized_params = jnp.array([], dtype=jnp.float32)
        except Exception as e:
             print(f"Error evaluating expression even without constants: {e}")
             return None
    else:
        print(f"Optimizing {num_constants} constants with DE (JITted loss, workers={de_workers})...")
        bounds = [constant_bounds] * num_constants

        try:
            # Note: `args=()` is implicit as loss_fn_for_scipy now gets its data via closure
            optimizer_result = differential_evolution(
                loss_fn_for_scipy,
                bounds,
                strategy=de_strategy,
                maxiter=de_maxiter,
                popsize=de_popsize_factor, # popsize = de_popsize_factor * N_dimensions
                tol=de_tol,
                mutation=de_mutation,
                recombination=de_recombination,
                polish=de_polish,
                disp=de_disp,
                workers=de_workers
            )

            if not optimizer_result.success:
                print(f"Warning: Differential Evolution did not report success! Message: {optimizer_result.message}")

            optimized_params = jnp.array(optimizer_result.x, dtype=jnp.float32)
            final_loss = optimizer_result.fun

            print(f"Optimization finished.")
            print(f"  Success (DE): {optimizer_result.success}")
            print(f"  Message (DE): {optimizer_result.message}")
            print(f"  Final Loss: {final_loss:.6f}")
            print(f"  Fitted Constants: {optimized_params}")
            print(f"  Number of function evaluations (NFEV): {optimizer_result.nfev}")

            if jnp.isnan(final_loss) or jnp.isinf(final_loss) or final_loss > 1e5:
                 print("Warning: Final loss is very high or invalid. The fit might be poor.")

        except Exception as e:
            print(f"Error during Differential Evolution optimization: {e}")
            test_params_np = np.array([ (constant_bounds[0] + constant_bounds[1]) / 2.0 ] * num_constants, dtype=np.float32)
            try:
                initial_loss_val = loss_fn_for_scipy(test_params_np) # Test with JITted version
                print(f"Loss with test params ({test_params_np}): {initial_loss_val}")
                if np.isnan(initial_loss_val): print("Loss is NaN even with test parameters.")
            except Exception as eval_e:
                print(f"Error evaluating JITted loss function even with test params: {eval_e}")
            return None

    # --- 6. Create and Return the Prediction Function ---
    # This function closes over the `optimized_params` and the RPN configuration
    # (_active_tokens_for_loss, _inv_vocab_for_loss, _epsilon_for_loss) and uses
    # the `evaluate_rpn_batch` defined in this scope.
    final_optimized_params = optimized_params # Capture from optimization block

    def prediction_function(new_x_coords):
        try:
            new_x_coords_jax = jnp.asarray(new_x_coords, dtype=jnp.float32)
            expected_ndim = new_x_coords_jax.ndim
            
            if expected_ndim == 1:
                if new_x_coords_jax.shape != (2,):
                    raise ValueError("Single input must be of shape (2,)")
                new_x_coords_batch = new_x_coords_jax.reshape(1, -1)
                y_pred_batch = evaluate_rpn_batch(
                    new_x_coords_batch, final_optimized_params,
                    _active_tokens_for_loss, _inv_vocab_for_loss, _epsilon_for_loss
                )
                return y_pred_batch[0]
            elif expected_ndim == 2:
                if new_x_coords_jax.shape[1] != 2:
                    raise ValueError("Batch input must be of shape (M, 2)")
                y_pred_batch = evaluate_rpn_batch(
                    new_x_coords_jax, final_optimized_params,
                    _active_tokens_for_loss, _inv_vocab_for_loss, _epsilon_for_loss
                )
                return y_pred_batch
            else:
                raise ValueError("Input must be a single coordinate (1D array/list of length 2) or a batch (2D array Nx2)")
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback NaN return based on detected input shape if possible
            if 'expected_ndim' in locals():
                 if expected_ndim == 1: return jnp.nan
                 elif expected_ndim == 2: return jnp.full(new_x_coords_jax.shape[0], jnp.nan)
            raise e # Re-raise if shape was truly unexpected or other error

    return prediction_function