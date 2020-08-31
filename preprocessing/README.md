## Preprocessing
We use MTCNN<sup>[CITE]</sup> for face detection and alignment. Each image should go through two passes. In the first pass, five points of the facial landmarks are detected and the rotation angle is computed. In the second pass, the rotated image is used to find the bounding box of the face. The rotation angle and the bounding box are saved in the csv files.

TODO: face alignment image