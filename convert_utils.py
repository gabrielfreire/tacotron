import tensorflowjs as tfjs
import tensorflow as tf

def make_saved_model (model_dir, export_dir, is_saved_model=False):
    # Load graph
#     g = Graph(mode="synthesize")
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if (not is_saved_model):
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        else:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
        # get graph node names to figure out which one is output
        l = [n.name for n in tf.get_default_graph().as_graph_def().node]
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
        builder.save()
        return l

def convert_to_tensorflowjs (model_dir, output_dir, output_node_names='Sigmoid'):
    tfjs.converters.convert_tf_saved_model(model_dir, output_node_names=output_node_names, output_dir=output_dir, 
                                            saved_model_tags='serve', skip_op_check=True, strip_debug_ops=True)