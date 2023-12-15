import math

import numpy as np
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


def create_X_y(
    X,
    y,
    bootstrap=True,
    split_perc=0.8,
    prob_type="regression",
    list_cont=None,
    random_state=None,
):
    """Create train/valid split of input data X and target variable y.
    Parameters
    ----------
    X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The input samples before the splitting process.
    y: ndarray, shape (n_samples, )
        The output samples before the splitting process.
    bootstrap: bool, default=True
        Application of bootstrap sampling for the training set.
    split_perc: float, default=0.8
        The training/validation cut for the provided data.
    prob_type: str, default='regression'
        A classification or a regression problem.
    list_cont: list, default=[]
        The list of continuous variables.
    random_state: int, default=2023
        Fixing the seeds of the random generator.

    Returns
    -------
    X_train_scaled: {array-like, sparse matrix}, shape (n_train_samples, n_features)
        The bootstrapped training input samples with scaled continuous variables.
    y_train_scaled: {array-like}, shape (n_train_samples, )
        The bootstrapped training output samples scaled if continous.
    X_valid_scaled: {array-like, sparse matrix}, shape (n_valid_samples, n_features)
        The validation input samples with scaled continuous variables.
    y_valid_scaled: {array-like}, shape (n_valid_samples, )
        The validation output samples scaled if continous.
    X_scaled: {array-like, sparse matrix}, shape (n_samples, n_features)
        The original input samples with scaled continuous variables.
    y_valid: {array-like}, shape (n_samples, )
        The original output samples with validation indices.
    scaler_x: scikit-learn StandardScaler
        The standard scaler encoder for the continuous variables of the input.
    scaler_y: scikit-learn StandardScaler
        The standard scaler encoder for the output if continuous.
    valid_ind: list
        The list of indices of the validation set.
    """
    rng = np.random.RandomState(random_state)
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    n = X.shape[0]

    if bootstrap:
        train_ind = rng.choice(n, n, replace=True)
    else:
        train_ind = rng.choice(n, size=int(np.floor(split_perc * n)), replace=False)
    valid_ind = np.array([ind for ind in range(n) if ind not in train_ind])

    X_train, X_valid = X[train_ind], X[valid_ind]
    y_train, y_valid = y[train_ind], y[valid_ind]

    # Scaling X and y
    X_train_scaled = X_train.copy()
    X_valid_scaled = X_valid.copy()
    X_scaled = X.copy()

    if len(list_cont) > 0:
        X_train_scaled[:, list_cont] = scaler_x.fit_transform(X_train[:, list_cont])
        X_valid_scaled[:, list_cont] = scaler_x.transform(X_valid[:, list_cont])
        X_scaled[:, list_cont] = scaler_x.transform(X[:, list_cont])
    if prob_type == "regression":
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_valid_scaled = scaler_y.transform(y_valid)
    else:
        y_train_scaled = y_train.copy()
        y_valid_scaled = y_valid.copy()

    return (
        X_train_scaled,
        y_train_scaled,
        X_valid_scaled,
        y_valid_scaled,
        X_scaled,
        y_valid,
        scaler_x,
        scaler_y,
        valid_ind,
    )


def sigmoid(x):
    """The function applies the sigmoid function element-wise to the input array x."""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """The function applies the softmax function element-wise to the input array x."""
    # Ensure numerical stability by subtracting the maximum value of x from each element of x
    # This prevents overflow errors when exponentiating large numbers
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def relu(x):
    """The function applies the relu function element-wise to the input array x."""
    return (abs(x) + x) / 2


def relu_(x):
    """The function applies the derivative of the relu function element-wise
    to the input array x.
    """
    return (x > 0) * 1


def convert_predict_proba(list_probs):
    """If the classification is done using a one-hot encoded variable, the list of
    probabilites will be a list of lists for the probabilities of each of the categories.
    This function takes the probabilities of having each category (=1 with binary) and stack
    them into one ndarray.
    """
    if len(list_probs.shape) == 3:
        list_probs = np.array(list_probs)[..., 1].T
    return list_probs


def ordinal_encode(y):
    """This function encodes the ordinal variable with a special gradual encoding storing also
    the natural order information.
    """
    list_y = []
    for y_col in range(y.shape[-1]):
        # Retrieve the unique values
        unique_vals = np.unique(y[:, y_col])
        # Mapping each unique value to its corresponding index
        mapping_dict = {}
        for i, val in enumerate(unique_vals):
            mapping_dict[val] = i + 1
        # create a zero-filled array for the ordinal encoding
        y_ordinal = np.zeros((len(y[:, y_col]), len(set(y[:, y_col]))))
        # set the appropriate indices to 1 for each ordinal value and all lower ordinal values
        for ind_el, el in enumerate(y[:, y_col]):
            y_ordinal[ind_el, np.arange(mapping_dict[el])] = 1
        list_y.append(y_ordinal[:, 1:])

    return list_y


def sample_predictions(predictions, random_state=None):
    """This function samples from the same leaf node of the input sample
    in both the regression and the classification case
    """
    rng = np.random.RandomState(random_state)
    return predictions[..., rng.randint(predictions.shape[2]), :]


def joblib_ensemble_dnnet(
    X,
    y,
    prob_type="regression",
    link_func=None,
    list_cont=None,
    list_grps=None,
    n_layer=5,
    bootstrap=False,
    split_perc=0.8,
    group_stacking=False,
    n_out_subLayers=1,
    n_hidden=None,
    n_epoch=200,
    batch_size=32,
    beta1=0.9,
    beta2=0.999,
    lr=1e-3,
    l1_weight=1e-2,
    l2_weight=1e-2,
    epsilon=1e-8,
    random_state=None,
):
    """
    Parameters
    ----------
    X : {array-like, sparse-matrix}, shape (n_samples, n_features)
        The input samples.
    y : {array-like}, shape (n_samples,)
        The output samples.
    random_state: int, default=None
        Fixing the seeds of the random generator
    """

    pred_v = np.empty(X.shape[0])
    # Sampling and Train/Validate splitting
    (
        X_train_scaled,
        y_train_scaled,
        X_valid_scaled,
        y_valid_scaled,
        X_scaled,
        y_valid,
        scaler_x,
        scaler_y,
        valid_ind,
    ) = create_X_y(
        X,
        y,
        bootstrap=bootstrap,
        split_perc=split_perc,
        prob_type=prob_type,
        list_cont=list_cont,
        random_state=random_state,
    )

    current_model = dnn_net(
        X_train_scaled,
        y_train_scaled,
        X_valid_scaled,
        y_valid_scaled,
        prob_type=prob_type,
        link_func=link_func,
        n_hidden=n_hidden,
        n_epoch=n_epoch,
        batch_size=batch_size,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        epsilon=epsilon,
        list_grps=list_grps,
        group_stacking=group_stacking,
        n_out_subLayers=n_out_subLayers,
        random_state=random_state,
    )

    if not group_stacking:
        X_scaled_n = X_scaled.copy()
    else:
        X_scaled_n = np.zeros((X_scaled.shape[0], len(list_grps) * n_out_subLayers))
        for grp_ind in range(len(list_grps)):
            X_scaled_n[
                :,
                list(
                    np.arange(
                        n_out_subLayers * grp_ind,
                        (grp_ind + 1) * n_out_subLayers,
                    )
                ),
            ] = (
                X_scaled[:, list_grps[grp_ind]].dot(current_model[3][grp_ind])
                + current_model[4][grp_ind]
            )

    for j in range(n_layer):
        if j == 0:
            pred = relu(X_scaled_n.dot(current_model[0][j]) + current_model[1][j])
        else:
            pred = relu(pred.dot(current_model[0][j]) + current_model[1][j])

    pred = pred.dot(current_model[0][n_layer]) + current_model[1][n_layer]

    if prob_type != "classification":
        if prob_type != "ordinal":
            pred_v = pred * scaler_y.scale_ + scaler_y.mean_
        else:
            pred_v = link_func[prob_type](pred)
        loss = np.std(y_valid) ** 2 - mean_squared_error(y_valid, pred_v[valid_ind])
    else:
        pred_v = link_func[prob_type](pred)
        loss = log_loss(
            y_valid, np.ones(y_valid.shape) * np.mean(y_valid, axis=0)
        ) - log_loss(y_valid, pred_v[valid_ind])

    return (current_model, scaler_x, scaler_y, pred_v, loss)


def dnn_net(
    X_train,
    y_train,
    X_valid,
    y_valid,
    prob_type="regression",
    link_func=None,
    n_hidden=None,
    n_epoch=200,
    batch_size=32,
    beta1=0.9,
    beta2=0.999,
    lr=1e-3,
    l1_weight=1e-2,
    l2_weight=1e-2,
    epsilon=1e-8,
    list_grps=None,
    group_stacking=False,
    n_out_subLayers=1,
    random_state=None,
):
    """
    Parameters
    ----------
    X_train : {array-like, sparse-matrix}, shape (n_samples, n_features)
        The input training samples.
    y_train : {array-like}, shape (n_samples,)
        The output training samples.
    X_valid : {array-like, sparse-matrix}, shape (n_samples, n_features)
        The input validation samples.
    y_valid : {array-like}, shape (n_samples,)
        The output validation samples.
    """
    n_layer = len(n_hidden)
    n, p = X_train.shape
    n_subLayers = len(list_grps)
    input_dim = (n_subLayers * n_out_subLayers) if group_stacking else p
    output_dim = y_train.shape[1]
    rng = np.random.RandomState(random_state)
    dropout_prob = 0.2

    loss = [None] * n_epoch
    best_loss = math.inf
    best_weight = []
    best_bias = []
    best_weight_stack = []
    best_bias_stack = []

    weight_stack = [None] * (n_subLayers)
    bias_stack = [None] * (n_subLayers)

    weight = [None] * (n_layer + 1)
    bias = [None] * (n_layer + 1)
    a = [None] * (n_layer + 1)
    h = [None] * (n_layer + 1)
    d_a = [None] * (n_layer + 1)
    d_h = [None] * (n_layer + 1)
    d_w = [None] * (n_layer + 1)
    dw = [None] * (n_layer + 1)
    db = [None] * (n_layer + 1)

    # ADAM
    mt_ind = 0
    mt_w_stack = [None] * (n_subLayers)
    vt_w_stack = [None] * (n_subLayers)
    mt_b_stack = [None] * (n_subLayers)
    vt_b_stack = [None] * (n_subLayers)

    mt_w = [None] * (n_layer + 1)
    vt_w = [None] * (n_layer + 1)
    mt_b = [None] * (n_layer + 1)
    vt_b = [None] * (n_layer + 1)

    dropout = [None] * n_layer
    if group_stacking:
        # Initialization of the stacking sub-layers
        # 1 sub-layer corresponds to 1 group
        for sub_layer in range(n_subLayers):
            # weight_stack[sub_layer] = rng.randn(len(list_grps[sub_layer]), n_out_subLayers) * np.sqrt(2/len(list_grps[sub_layer]))
            # bias_stack[sub_layer] = rng.randn(1, n_out_subLayers) * np.sqrt(2/1)
            weight_stack[sub_layer] = (
                rng.uniform(-1, 1, (len(list_grps[sub_layer]), n_out_subLayers)) * 0.1
            )
            bias_stack[sub_layer] = rng.uniform(-1, 1, (1, n_out_subLayers)) * 0.05
            # weight_stack[sub_layer] = rng.normal(0, 1, (len(list_grps[sub_layer]), n_out_subLayers)) * 0.1
            # bias_stack[sub_layer] = rng.normal(-1, 1, (1, n_out_subLayers)) * 0.05
            mt_w_stack[sub_layer] = np.zeros(
                (len(list_grps[sub_layer]), n_out_subLayers)
            )
            mt_b_stack[sub_layer] = np.zeros((1, n_out_subLayers))
            vt_w_stack[sub_layer] = np.zeros(
                (len(list_grps[sub_layer]), n_out_subLayers)
            )
            vt_b_stack[sub_layer] = np.zeros((1, n_out_subLayers))

    # Initialization of the remaining layers
    for i in range(n_layer + 1):
        if i == 0:
            # weight[i] = rng.randn(input_dim, n_hidden[i]) * np.sqrt(2/input_dim)
            # bias[i] = rng.randn(1, n_hidden[i]) * np.sqrt(2/1)
            weight[i] = rng.uniform(-1, 1, (input_dim, n_hidden[i])) * 0.1
            bias[i] = rng.uniform(-1, 1, (1, n_hidden[i])) * 0.05
            mt_w[i] = np.zeros((input_dim, n_hidden[i]))
            mt_b[i] = np.zeros((1, n_hidden[i]))
            vt_w[i] = np.zeros((input_dim, n_hidden[i]))
            vt_b[i] = np.zeros((1, n_hidden[i]))
        elif i == n_layer:
            # weight[i] = rng.randn(n_hidden[i-1], output_dim) * np.sqrt(2/n_hidden[i-1])
            # bias[i] = rng.randn(1, output_dim) * np.sqrt(2/1)
            weight[i] = rng.uniform(-1, 1, (n_hidden[i - 1], output_dim)) * 0.1
            bias[i] = rng.uniform(-1, 1, (1, output_dim)) * 0.05
            mt_w[i] = np.zeros((n_hidden[i - 1], output_dim))
            mt_b[i] = np.zeros((1, output_dim))
            vt_w[i] = np.zeros((n_hidden[i - 1], output_dim))
            vt_b[i] = np.zeros((1, output_dim))
        else:
            # weight[i] = rng.randn(n_hidden[i-1], n_hidden[i]) * np.sqrt(2/n_hidden[i-1])
            # bias[i] = rng.randn(1, n_hidden[i]) * np.sqrt(2/1)
            weight[i] = rng.uniform(-1, 1, (n_hidden[i - 1], n_hidden[i])) * 0.1
            bias[i] = rng.uniform(-1, 1, (1, n_hidden[i])) * 0.05
            mt_w[i] = np.zeros((n_hidden[i - 1], n_hidden[i]))
            mt_b[i] = np.zeros((1, n_hidden[i]))
            vt_w[i] = np.zeros((n_hidden[i - 1], n_hidden[i]))
            vt_b[i] = np.zeros((1, n_hidden[i]))

    # Getting the batches ready
    n_round = int(np.ceil(n / batch_size))
    i_bgn = np.empty(n_round, dtype=np.int32)
    i_end = np.empty(n_round, dtype=np.int32)

    for s in range(n_round):
        i_bgn[s] = s * batch_size
        i_end[s] = min((s + 1) * batch_size, n)

    for k in range(n_epoch):
        # Shuffling the instances
        new_order = rng.choice(n, n, replace=False)
        X_train_n = X_train[new_order, :]
        y_train_n = y_train[new_order]

        for i in range(n_round):
            # Going through the different batches
            xi = X_train_n[i_bgn[i] : i_end[i], :]
            yi = y_train_n[i_bgn[i] : i_end[i]]

            if not group_stacking:
                xi_n = xi.copy()
            else:
                xi_n = np.zeros((xi.shape[0], n_subLayers * n_out_subLayers))
                # Linear sub-layers without activation (Stacking groups)
                for grp_ind in range(len(list_grps)):
                    xi_n[
                        :,
                        list(
                            np.arange(
                                n_out_subLayers * grp_ind,
                                (grp_ind + 1) * n_out_subLayers,
                            )
                        ),
                    ] = (
                        xi[:, list_grps[grp_ind]].dot(weight_stack[grp_ind])
                        + bias_stack[grp_ind]
                    )
            # Forward propagation
            for j in range(n_layer):
                if j == 0:
                    a[j] = xi_n.dot(weight[j]) + bias[j]
                else:
                    a[j] = h[j - 1].dot(weight[j]) + bias[j]
                # Inverted dropout
                # dropout[j] = rng.binomial(1, 1 - dropout_prob, a[j].shape) / (1 - dropout_prob)
                # a[j] *= dropout[j]
                h[j] = relu(a[j])

            y_pi = h[n_layer - 1].dot(weight[n_layer]) + bias[n_layer]

            if prob_type != "regression":
                y_pi = link_func[prob_type](y_pi)

            # Backward propagation
            d_a[n_layer] = -(yi - y_pi) / len(yi)
            d_w[n_layer] = h[n_layer - 1].T.dot(d_a[n_layer])

            mt_ind += 1
            for j in range(n_layer, -1, -1):
                if j != n_layer:
                    # if j == 0:
                    #     d_h[j] = d_a[j+1].dot(weight[j+1].T)
                    # else:
                    d_h[j] = d_a[j + 1].dot(weight[j + 1].T)
                    # d_h[j] *= dropout[j]
                    d_a[j] = d_h[j] * relu_(a[j])
                    if j > 0:
                        d_w[j] = h[j - 1].T.dot(d_a[j])
                    else:
                        d_w[j] = xi_n.T.dot(d_a[j])
                bias_grad = np.ones((d_a[j].shape[0], 1)).T.dot(d_a[j])
                mt_w[j] = mt_w[j] * beta1 + (1 - beta1) * d_w[j]
                mt_b[j] = mt_b[j] * beta1 + (1 - beta1) * bias_grad
                vt_w[j] = vt_w[j] * beta2 + (1 - beta2) * d_w[j] ** 2
                vt_b[j] = vt_b[j] * beta2 + (1 - beta2) * bias_grad**2
                dw[j] = (
                    lr
                    * mt_w[j]
                    / (1 - beta1**mt_ind)
                    / (np.sqrt(vt_w[j] / (1 - beta2**mt_ind)) + epsilon)
                )
                db[j] = (
                    lr
                    * mt_b[j]
                    / (1 - beta1**mt_ind)
                    / (np.sqrt(vt_b[j] / (1 - beta2**mt_ind)) + epsilon)
                )

                weight[j] = (
                    weight[j]
                    - dw[j]
                    - l1_weight * ((weight[j] > 0) * 1 - (weight[j] < 0) * 1)
                    - l2_weight * weight[j]
                )
                bias[j] = bias[j] - db[j]

            if group_stacking:
                d_h_stack = d_a[0].dot(weight[0].T)
                # The derivative of the identity function is 1
                d_a_stack = d_h_stack * 1

                bias_grad = np.ones((d_a_stack.shape[0], 1)).T.dot(d_a_stack)
                # Including groups stacking in backward propagation
                for grp_ind in range(len(list_grps)):
                    d_w_stack = xi[:, list_grps[grp_ind]].T.dot(d_a_stack[:, [grp_ind]])

                    mt_w_stack[grp_ind] = (
                        mt_w_stack[grp_ind] * beta1 + (1 - beta1) * d_w_stack
                    )
                    mt_b_stack[grp_ind] = (
                        mt_b_stack[grp_ind] * beta1
                        + (1 - beta1) * bias_grad[:, [grp_ind]]
                    )
                    vt_w_stack[grp_ind] = (
                        vt_w_stack[grp_ind] * beta2 + (1 - beta2) * d_w_stack**2
                    )
                    vt_b_stack[grp_ind] = (
                        vt_b_stack[grp_ind] * beta2
                        + (1 - beta2) * bias_grad[:, [grp_ind]] ** 2
                    )

                    dw_stack = (
                        lr
                        * mt_w_stack[grp_ind]
                        / (1 - beta1**mt_ind)
                        / (
                            np.sqrt(vt_w_stack[grp_ind] / (1 - beta2**mt_ind))
                            + epsilon
                        )
                    )
                    db_stack = (
                        lr
                        * mt_b_stack[grp_ind]
                        / (1 - beta1**mt_ind)
                        / (
                            np.sqrt(vt_b_stack[grp_ind] / (1 - beta2**mt_ind))
                            + epsilon
                        )
                    )

                    weight_stack[grp_ind] = (
                        weight_stack[grp_ind]
                        - dw_stack
                        - l1_weight
                        * (
                            (weight_stack[grp_ind] > 0) * 1
                            - (weight_stack[grp_ind] < 0) * 1
                        )
                        - l2_weight * weight_stack[grp_ind]
                    )
                    bias_stack[grp_ind] = bias_stack[grp_ind] - db_stack

        if not group_stacking:
            X_valid_n = X_valid.copy()
        else:
            X_valid_n = np.zeros((X_valid.shape[0], n_subLayers * n_out_subLayers))
            for grp_ind in range(len(list_grps)):
                X_valid_n[
                    :,
                    list(
                        np.arange(
                            n_out_subLayers * grp_ind,
                            (grp_ind + 1) * n_out_subLayers,
                        )
                    ),
                ] = (
                    X_valid[:, list_grps[grp_ind]].dot(weight_stack[grp_ind])
                    + bias_stack[grp_ind]
                )

        for j in range(n_layer):
            if j == 0:
                pred = relu(X_valid_n.dot(weight[j]) + bias[j])
            else:
                pred = relu(pred.dot(weight[j]) + bias[j])

        pred = pred.dot(weight[n_layer]) + bias[n_layer]

        if prob_type != "regression":
            pred = link_func[prob_type](pred)

        if prob_type == "classification":
            loss[k] = log_loss(y_valid, pred)
        else:
            loss[k] = mean_squared_error(y_valid, pred)

        if loss[k] < best_loss:
            best_loss = loss[k]
            best_weight = weight.copy()
            best_bias = bias.copy()
            best_weight_stack = weight_stack.copy()
            best_bias_stack = bias_stack.copy()

    return [
        best_weight,
        best_bias,
        best_loss,
        best_weight_stack,
        best_bias_stack,
    ]
