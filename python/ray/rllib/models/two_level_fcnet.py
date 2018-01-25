from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models.model import Model
from ray.rllib.models.fcnet import FullyConnectedNetwork

MODEL_CONFIGS = [
    # === Required options ===
    "num_subpolicies",  # Number of subpolicies in two-level fcnet
    "hierarchical_fcnet_hiddens",  # Num. of hidden layers for two-level fcnet
    # === Other options ===
    "switching_fcnet_hiddens",  # Number of hidden layers for switching network
    # function which maps from observation to subpolicy observation
    "fn_subpolicy_state",
    # function which maps from observation to choice of subpolicy
    "fn_choose_subpolicy",
]


class TwoLevelFCNetwork(Model):
    """
    Two-level fully connected network, consisting of a number of
    subpolicies and a pre-specified function which chooses among the
    subpolicies.
    """

    def _init(self, inputs, num_outputs, options):
        custom_options = options["custom_options"]
        subhiddens = custom_options.get("hierarchical_fcnet_hiddens",
                                        [[256, 256]] * 1)

        print("Constructing two level fcnet {}".format(subhiddens))
        num_subpolicies = custom_options.get("num_subpolicies", 1)
        # function which maps from observation to subpolicy observation
        to_subpolicy_state = custom_options.get("fn_subpolicy_state", None)
        # function which maps from observation to choice of subpolicy
        fn_choose_policy = custom_options.get("fn_choose_subpolicy", None)

        if to_subpolicy_state is None:
            to_subpolicy_state = lambda x, k: x
        else:
            # defines to_subpolicy_state function
            eval(compile(to_subpolicy_state, '<string>', 'exec'), globals())
            to_subpolicy_state = globals()['to_subpolicy_state']

        if fn_choose_policy is None:
            # choose_policy = lambda x: x
            # choose_policy = lambda x: tf.cast(x[:, 7] > 210, tf.int32)
            switching_hiddens = custom_options.get("switching_fcnet_hiddens",
                                                   [32, 32])
            attn_options = {"fcnet_hiddens": switching_hiddens}
            attn_options["user_data"] = {"fcnet_tag": 'attn'}
            # TODO add option to specify network hiddens
            # TODO add option to one-hot / max the vector
            fcnet = FullyConnectedNetwork(
                inputs, num_subpolicies, attn_options)
            attention = fcnet.outputs
        else:
            # defines choose_policy function
            eval(compile(fn_choose_policy, '<string>', 'exec'), globals())
            choose_policy = globals()['choose_policy']
            attention = tf.one_hot(choose_policy(inputs), num_subpolicies)

        outputs = []
        for k in range(num_subpolicies):
            with tf.variable_scope("multi{}".format(k)):
                sub_options = options.copy()
                sub_options.update({"fcnet_hiddens": subhiddens[k]})
                subinput = to_subpolicy_state(inputs, k)
                fcnet = FullyConnectedNetwork(
                    subinput, num_outputs, sub_options)
                output = fcnet.outputs
                rep_attention = tf.reshape(tf.tile(attention[:, k],
                                          [num_outputs]),
                                          [-1, num_outputs])
                outputs.append(rep_attention * output)
        overall_output = tf.add_n(outputs)
        # TODO(cathywu) check that outputs is not used later on because it's
        # a list instead of a layer
        return overall_output, outputs
