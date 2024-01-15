# Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection in PyTorch

This code is based out of the paper [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection.](https://openreview.net/pdf?id=BJJLHbb0-)

I used baseline code from this repo  [tnakae implementation](https://github.com/tnakae/DAGMM) which was implemented in tensorflow 1.x. Thanks.

Later I modified the code to temsorflow 2.x version.

Please Let me know if there are any bugs in my code. Thank you! =)

Myself implemented this code version in Tensorflow 2.13 version using Python 3.8

## Execution
```python
  ## Please change the parameters according to your case
  ## If you want to change core parameters default values, then you need to tweak in DAGMM.py

  dagmm_model = DAGMM(comp_hiddens=[16,8,1], comp_activation=tf.nn.tanh,
                      est_hiddens=[8,4], est_activation=tf.nn.tanh)

  # Train the model
  dagmm_model.fit(model_input_data)

  # Model Prediction
  model_pred = dagmm.predict(model_input_data)

  # Specify the folder to save plots
  save_folder = 'dagmm_plots'

  # Create the folder if it doesn't exist
  if not os.path.exists(save_folder):
      os.makedirs(save_folder)

  #### Energy distribution plot
  plt.figure(figsize=[8,3])
  plt.plot(model_pred, "o-")
  plt.xlabel("Index (row) of Sample")
  plt.ylabel("Energy")
  # Save the figure in the specified folder
  save_path = os.path.join(save_folder, 'energy_plot.png')
  plt.savefig(save_path)

  #### In my case, I had 18 features, so I made 6x3 subplots.
  fig, axes = plt.subplots(nrows=6, ncols=3, figsize=[12,12], sharex=True, sharey=True)
  plt.subplots_adjust(wspace=0.05, hspace=0.05)

  # Here, we have 18 input features, so 6*3
  for row in range(6):
      for col in range(3):
          ax = axes[row, col]
          if row != col:
              ax.plot(model_input_data[:,col], model_input_data[:,row], ".")
              ano_index = np.arange(len(model_pred))[model_pred > np.percentile(model_pred, 99)]
              ax.plot(model_input_data[ano_index,col], model_input_data[ano_index,row], "x", c="r", markersize=8)

  # Save the figure in the specified folder
  save_path = os.path.join(save_folder, 'dagmm_scatter_plots.png')
  plt.savefig(save_path)
```

