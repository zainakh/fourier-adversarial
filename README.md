# fourier-adversarial

This project displays the vulnerability of reconstruction networks to adversarial perturbations. This is shown through direct comparison of three different cases. The first is using an reconstruction network with adversarial perturbations applied to the input. It is constructed to give the maximum SSE, relative to the original full resolution MRI image, in reconstruction via the Fast Gradient Sign Method. The second uses Gaussian perturbations in place of the adversarial noise for comparison. Finally, these are compared to a "perfect" reconstruction: the Fourier transform. Any perturbations made in the frequency domain will be preserved, with the addition of a scaling factor, when "reconstructing" into the image space. These three cases are run across a variety of perturbation sizes on a knee MRI scan. 

To run the main script, run using the following command line when in the base directory of the project:
```
python corruptFourier.py -f *name of file to save data to* -i *name of new file you want to save image to* -fc
```

The `-fc` flag at the end means you run with a full set of Fourier coefficients. You can replace this with `-p` to run with a partial set of coefficients. The data that is saved can be plotted via the `plotResults.py` file. 
