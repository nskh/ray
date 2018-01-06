from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models.model import Model
from ray.rllib.models.fcnet import FullyConnectedNetwork

USER_DATA_CONFIGS = [
    "num_subpolicies",  # Number of subpolicies for a two-level network
    "fn_choose_subpolicy",  # Function for choosing subpolicy
    "fn_subpolicy_state",  # Function for mapping observations to subpolicy obs
]


class TwoLevelFCNetwork(Model):
    """
    Two-level fully connected network, consisting of a number of
    subpolicies and a pre-specified function which chooses among the
    subpolicies.
    """

    def _init(self, inputs, num_outputs, options):
        subhiddens = options.get("fcnet_hiddens", [[256, 256]] * 1)
        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu
        print("Constructing two level fcnet {} {}".format(subhiddens,
                                                         activation))

        user_data = options.get("user_data", {})
        for k in user_data.keys():
            if k not in USER_DATA_CONFIGS:
                raise Exception(
                    "Unknown config key `{}`, all keys: {}".format(k,
                                                            USER_DATA_CONFIGS))
        num_subpolicies = user_data.get("num_subpolicies", 1)
        # function which maps from observation to subpolicy observation
        to_subpolicy_state = user_data.get("fn_subpolicy_state", None)
        # function which maps from observation to choice of subpolicy
        fn_choose_policy = user_data.get("fn_choose_subpolicy", None)

        if to_subpolicy_state is None:
            to_subpolicy_state = lambda x, k: x
        else:
            # defines to_subpolicy_state function
            eval(compile(to_subpolicy_state, '<string>', 'exec'), globals())
            to_subpolicy_state = globals()['to_subpolicy_state']

        if fn_choose_policy is None:
            # choose_policy = lambda x: x
            choose_policy = lambda x: tf.cast(x[:, 7] > 210, tf.int32)
        else:
            # defines choose_policy function
            eval(compile(fn_choose_policy, '<string>', 'exec'), globals())
            choose_policy = globals()['choose_policy']

        attention = tf.one_hot(choose_policy(inputs), num_subpolicies)

        outputs = []
        for k in range(num_subpolicies):
            sub_options = options.copy()
            sub_options.update({"fcnet_hiddens": subhiddens[k]})
            sub_options["user_data"] = {"fcnet_tag": k}
            subinput = to_subpolicy_state(inputs, k)
            fcnet = FullyConnectedNetwork(
                subinput, num_outputs, sub_options)
            output, last_layer = fcnet.outputs, fcnet.last_layer
            rep_attention = tf.reshape(tf.tile(attention[:, k], [num_outputs]),
                [-1, num_outputs])
            outputs.append(rep_attention * output)
        overall_output = tf.add_n(outputs)
        # TODO(cathywu) check that outputs is not used later on because it's
        # a list instead of a layer
        return overall_output, outputs
