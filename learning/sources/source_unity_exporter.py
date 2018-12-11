import tensorflow as tf
import tensorblock as tb
from tensorflow.python.tools import freeze_graph


def export_ugraph( brain, model_path, env_name, target_nodes):
    """
    Unity ML Agents
    Exports latest saved model to .bytes format for Unity embedding.
    :brain: tensorblock brain
    :param model_path: path of model checkpoints.
    :param env_name: Name of associated Learning Environment.
    :param target_nodes: Comma separated string of needed output nodes for embedded graph.
    Example: To export: from sources.source_unity_exporter import *
                        export_ugraph (self.brain, "./trained_models/unity_contcatch_player_DDPG/", "continuouscatcher", "NormalActor/Output/Tanh")
                        raise SystemExit(0)
             On Unity: scope = NormalActor/
                       action = /Output/Tanh
                       observation = Observation/Placeholder
    """
    tf.train.write_graph(tf.Session().graph_def, model_path, 'raw_graph_def.pb', as_text=False)
    ckpt = tf.train.get_checkpoint_state(model_path)
    freeze_graph.freeze_graph(input_graph=model_path + '/raw_graph_def.pb',
                              input_binary=True,
                              input_checkpoint=ckpt.model_checkpoint_path,
                              output_node_names=target_nodes,
                              output_graph=model_path + '/' + env_name + '.bytes',
                              clear_devices=True, initializer_nodes="", input_saver="",
                              restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")
