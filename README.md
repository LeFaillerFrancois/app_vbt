![logo](read_me_preview/Logo.png)
<br>
A tkinter app on pc to track barbell speed and trajectory in powerlifting. Can also compute Load-Velocity Profile / Force Velocity Profile.

## Showcase
*The returned video*
<br>
![returned video](read_me_preview/exemple_gif.gif)
<br>
*The Main Menu*
<br>
![main_menu](read_me_preview/main_menu.png)
<br>
*The single file analysis page : Range of Motion over Reps plot, Mean Speed over Reps plot, Trajectory plot*
<br>
![single_file_analysis_showcase](read_me_preview/single_file_analysis_showcase.png)
<br>
*The mutiple file analysis page : Force-Velocity Profiling*
<br>
![multiple_file_analysis_showcase](read_me_preview/multiple_file_fv_profile_showcase.png)
<br>
*Possibility to manually edit the rep automatic detection in case it didn't work as intended*
<br>
![rep_editor](read_me_preview/rep_editor_showcase.png)
<br>

## How it works 
Using DeepLabCut, I've created my own light model to track the center of a barbell as well as the top and the bottom of a plate. Knowing that a calibrated plate of 20kg + is 45cm, it is then possible to convert pixels in cm and to measure speed.
If you are not using calibrated weight, you could still change the value "plate size" to the corresponding one.

## How to use it
First get a portrait side view video of any powerlifting lift with calibrated plates. Download [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and create an python env with it.
We are compressing the quality of the original video to analyse faster. 
Videos and csv files will be created in the folder wich contains your original video.

## Limitations
Perfectly sideview is necessary for accurate bar path, otherwise only bar speed will be usable (and only if the camera wasnt tilted). 
Be carefull using it to not have other plates in the camera range, could cause tracking issues.  

### In coming soon
- Better read me (this one is temporary)
- More metrics
- Downloadable datas and plots
- Better live webcam mode (Work in progress)
- ...
