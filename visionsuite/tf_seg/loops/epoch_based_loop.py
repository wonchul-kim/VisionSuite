import tensorflow as tf 
import time 

# @tf.function(experimental_relax_shapes=True)#(jit_compile=True) # XLA 컴파일 활성화
def train_step(model, inputs, optimizer, loss_fn, strategy, train_acc_metric):
    def step_fn(images, masks):
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            per_example_loss = loss_fn(tf.squeeze(masks, -1), preds)
            loss = tf.nn.compute_average_loss(
                    per_example_loss,
                    global_batch_size=2*strategy.num_replicas_in_sync
                )

            loss += sum(model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_acc_metric.update_state(tf.squeeze(masks, -1), preds)
        
        return loss

    images, masks = inputs
    per_replica_loss = strategy.run(step_fn, args=(images, masks))
    
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

def train_loop(model, dataset, epochs, optimizer, loss_fn, strategy, train_acc_metric, logs_dir=None):
    if logs_dir:
        tf.profiler.experimental.start(logdir=logs_dir)

    for epoch in range(epochs):
        total_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        gpu = f'{tf.config.experimental.get_memory_info("GPU:0")["current"]/1e7:.2f}GB'
        tic = time.time()
        for batch_idx, batch in enumerate(dataset):
            loss = train_step(model, batch, optimizer, loss_fn, strategy, train_acc_metric)
            total_loss += loss
            num_batches += 1
            
            if batch_idx%100 == 0:
                print(">>>>> time for epoch: ", time.time() - tic)
                tic = time.time()

            print(f'\r{epoch}/{epochs} > {batch_idx}: loss({loss:.4f}), acc({epoch_acc:.4f}), gpu({gpu})', end='', flush=True)
            
        epoch_loss = total_loss / num_batches
        epoch_acc = train_acc_metric.result()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        train_acc_metric.reset_states()
        
    if logs_dir:
        tf.profiler.experimental.stop()


