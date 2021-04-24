# AeroVision

## Intoduction
The main drivers in today's aircraft design are performance improvements, reduction of fuel consumption and harmful emission.  
These goals can be achieved in a straightforward fashion with lightweight, large wingspan designs. 
However, such configurations are inherently more flexible and susceptible to adverse aeroelastic phenomena including reduced control authority, increased maneuver loads, excessive response to atmospheric turbulence, and flutter instability. 
The immediate remedy for aeroelastic problems is stiffening the structure.  
This, however, comes at the cost of additional structural weight and, ultimately, a penalty to the systems performance.
Over the years, and with advances in aircraft control methodologies,  studies have shown that a lightweight, flexible air vehicle can be controlled to mitigate flutter instability, alleviate gust response, or
achieve optimal performance while minimizing adverse aeroelastic effects.
Most of the commonly used means to control aeroelastic phenomena rely on manipulating the aerodynamic forces achieved by static or dynamic activation of control surfaces. 
The control surfaces effectively modify the flexible wings' deformed shape, such that the resulting flow regime and pressure distribution generate favorable aerodynamic forces. 
However, as of yet, direct measurement of the deformed shape of a wing is not available. 
Therefore, active control systems provide feedback based on local measurement of acceleration (e.g. wingtip acceleration), or strain (e.g., a strain-gauge at the wing root), but not directly on displacements. 
Some numerical studies on aeroelastic control assume that the deformed shape is known. 
It could be in the form of wingtip displacement, or modal deformation. 
However, neither is readily measurable. 
It is noted that local deformations can be obtained from accelerometer measurements by temporal integration. 
Still, this information is local and limited to a few points, and the double temporal integration of accelerations to obtain deformations introduces errors.
Recent studies suggested that a detailed deformed shape of a structure could be obtained from strain-data measured in optical fibers. 
Fiber-optic sensing is used in civil engineering, aerospace, marine, and oil and gas, and their inherent capabilities make them highly suitable for embedding in aerospace systems and specifically in wings.    While Fiber-optic sensors (FOS)  appear to offer accurate information on wing deformations, their usage is not suitable for small air vehicles. The smallest FOS interrogators weighs approximately 1.5 kg, making for a bulky, heavy, and expensive system to fly. 
The primary goal of this research is to explore whether we can extract accurate information on the deformations of a flexible wing in flight using inexpensive, lightweight, off-the-shelf cameras.  
The study proposes developing a methodology based on Deep Learning, which receives as input images of a deformed wing, as captured by wing-mounted cameras, and produces, as an output, a set of parameters defining the deformed wing shape. The research targets to test the methodology numerically and experimentally using a wind-tunnel model of a flexible wing that has dynamical properties typical to those of realistic aircraft wings. 
If proven viable, such camera systems can be carried on-board lightweight vehicles, providing vital information for controlling and mitigating aeroelastic phenomena. This will lead to ever-lighter configurations, better performance, and lower environmental impact, which rely on active-control based on direct deformation measurement. 
In this study, we make an important first step - we implement and test the proposed methodology using synthetic, computational data. We simulate a large dataset of 2D wing images, along with their corresponding deformation parameters and train a neural network, validating it on many unseen examples. We show very promising results in preparation for the experimental phase where this methodology will be empirically tested under lab conditions. 

## File Structure
1. Under */docs/* You will find the full report, all .tex code, the kickoff presentation and the final presentation.
2. Under */src/* you will find all relevant source code, this is further expanded in Usage.
3. Under */data/* you will find all relevant data necessary for generating the training database, including textures used, the validation split in np format, all .off files, etc.
4. */data/data_samples/Matlab.rar* includes all Matlab code necessary for generating the raw wing displacement data, used later in the database generation.

## Usage
All relevant ready-to-run code is located in the /src/tests folder.
- */src/tests/test_database.py* includes the main function for generating the database used for training from the data supplied 
by the Aeroelasticity Lab in the Technion, a default config is supplied, batch size may be changed for performance, depending on available ram.
- */src/tests/test_resnet_synth.py* includes the main function(*test_resnet_noisy*) used for training the network,
  before running it should be configured with the specific DNN to be used, batch size, additional added Poisson/Gaussian/S&P noise, num input/output layers, experiment name, database path and validation/training split.
  
