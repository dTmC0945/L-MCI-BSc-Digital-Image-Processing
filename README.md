
# Table of Contents

1.  [A Gentle Introduction](#org6bdbe1a)
    1.  [Lecture Information](#orga66fad8)
        1.  [Assignments](#org8ab78d2)
    2.  [The Lecture Structure](#org6649915)
    3.  [Code Supplement](#org1715c47)
    4.  [Reading List](#orgddf167e)


<a id="org6bdbe1a"></a>

# A Gentle Introduction

Welcome to the lecture materials for use in **B.Sc - Digital Image Processing** where our
focus will be on the topics of:

1.  Fundamentals on discrete mathematics,
2.  Display technologies,
3.  Image processing techniques,
4.  An example of using ML in image recognition techniques.


<a id="orga66fad8"></a>

## Lecture Information

The details of the lecture are given below.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">DESCRIPTION</th>
<th scope="col" class="org-left">VALUE</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Program Name</td>
<td class="org-left">Bachelor's program "Mechatronics Design and Innovation"</td>
</tr>


<tr>
<td class="org-left">Module Name</td>
<td class="org-left">Image Processing</td>
</tr>


<tr>
<td class="org-left">Semester</td>
<td class="org-left">5</td>
</tr>


<tr>
<td class="org-left">Room</td>
<td class="org-left">Lecture Room</td>
</tr>


<tr>
<td class="org-left">Assessment(s)</td>
<td class="org-left">Individual Assignment (40 %) Group Assignment (60 %)</td>
</tr>


<tr>
<td class="org-left">Lecturer</td>
<td class="org-left">Daniel McGuiness</td>
</tr>


<tr>
<td class="org-left">Software</td>
<td class="org-left">Python</td>
</tr>


<tr>
<td class="org-left">Hardware</td>
<td class="org-left">-</td>
</tr>


<tr>
<td class="org-left">SWS Total</td>
<td class="org-left">4</td>
</tr>


<tr>
<td class="org-left">Total Units</td>
<td class="org-left">60</td>
</tr>


<tr>
<td class="org-left">ECTS</td>
<td class="org-left">5</td>
</tr>


<tr>
<td class="org-left">Lecture Type</td>
<td class="org-left">ILV</td>
</tr>
</tbody>
</table>


<a id="org8ab78d2"></a>

### Assignments

There will be two (**2**) assignments for this course.

The grade breakdown is as follows:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">DEFINITION</th>
<th scope="col" class="org-right">GRADE (%)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Individual Assignment</td>
<td class="org-right">40</td>
</tr>


<tr>
<td class="org-left">Group Assignment</td>
<td class="org-right">60</td>
</tr>


<tr>
<td class="org-left">Sum</td>
<td class="org-right">100</td>
</tr>
</tbody>
</table>

**Individual Assignment**

An individual assignment will be given to you to work on. This assignment will consist of
questions pertaining to concepts and image processing techniques.

The grade breakdown is as follows:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">DEFINITION</th>
<th scope="col" class="org-right">GRADE (%)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Report Style</td>
<td class="org-right">15</td>
</tr>


<tr>
<td class="org-left">Q1 - Blurring Filters</td>
<td class="org-right">15</td>
</tr>


<tr>
<td class="org-left">Q2 - Image Channel Analysis</td>
<td class="org-right">10</td>
</tr>


<tr>
<td class="org-left">Q3 - RNG Map Generation</td>
<td class="org-right">10</td>
</tr>


<tr>
<td class="org-left">Q4 - Image Cleaning</td>
<td class="org-right">10</td>
</tr>


<tr>
<td class="org-left">Q5 - Shape Recognition</td>
<td class="org-right">30</td>
</tr>


<tr>
<td class="org-left">Q6 - Image Quality Comparison</td>
<td class="org-right">10</td>
</tr>


<tr>
<td class="org-left">Sum</td>
<td class="org-right">100</td>
</tr>
</tbody>
</table>

**NOTE:** The assignment is individual and is not meant to be worked as a group. Once the
code and the work is submitted it will be vetted against a software to determine
if any collusion has occured.

**Group Assignment**

The group assignment focuses on a student defined project which its presentation will be
done in the last 3 sessions of the course. You are to come up with a group and a project
within the first 3 weeks of the lecture otherwise one will be given to you.

The grade breakdown is as follows:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">DEFINITION</th>
<th scope="col" class="org-right">GRADE (%)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Report Style</td>
<td class="org-right">15</td>
</tr>


<tr>
<td class="org-left">Content</td>
<td class="org-right">55</td>
</tr>


<tr>
<td class="org-left">Q &amp; A</td>
<td class="org-right">30</td>
</tr>
</tbody>
</table>

In report writing students must declare their contribution to the work and they will be
asked regarding their field of work during the Q&A (i.e., if Student A has worked with
blurring filter he may be asked on why a specific one is chosen and/or the concepts and
maths behind the said filter).

**NOTE:** Students will be graded based on their contribution to the project and answers
during the Q&A, therefore will be graded individually.


<a id="org6649915"></a>

## The Lecture Structure

As it currently is, the lecture covers topic from vision technologies (i.e., camera, display) to
methods in improving/analysing gathered images. The structure of the lecture is shown below.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-right">ORDER</th>
<th scope="col" class="org-left">TOPIC</th>
<th scope="col" class="org-left">DESCRIPTION</th>
<th scope="col" class="org-right">SESSION</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-right">1</td>
<td class="org-left">Introduction</td>
<td class="org-left">Discussion of the lecture structure and what will be covered</td>
<td class="org-right">1</td>
</tr>


<tr>
<td class="org-right">2</td>
<td class="org-left"><a href="file:///Users/danielmcguiness/GitHub/MCI-Source-Files/(BSc - Lecture) Digital Image Processing/Lecture Slides/DIP Slide/codes/Mathematical-Fundamentals.html">Mathematical Fundamentals</a></td>
<td class="org-left">Convolution, sampling theorem and Fourier analysis</td>
<td class="org-right">1</td>
</tr>


<tr>
<td class="org-right">3</td>
<td class="org-left"><a href="file:///Users/danielmcguiness/GitHub/MCI-Source-Files/(BSc - Lecture) Digital Image Processing/Lecture Slides/DIP Slide/codes/Perception.html">Perception</a></td>
<td class="org-left">Colour spaces and industry standards (i.e., colour science)</td>
<td class="org-right">2</td>
</tr>


<tr>
<td class="org-right">4</td>
<td class="org-left">Camera</td>
<td class="org-left">Camera operation principles and lenses</td>
<td class="org-right">2 - 3</td>
</tr>


<tr>
<td class="org-right">5</td>
<td class="org-left">Display</td>
<td class="org-left">Display technologies and standards</td>
<td class="org-right">4</td>
</tr>


<tr>
<td class="org-right">6</td>
<td class="org-left"><a href="file:///Users/danielmcguiness/GitHub/MCI-Source-Files/(BSc - Lecture) Digital Image Processing/Lecture Slides/DIP Slide/codes/Noise.corg">Noise</a></td>
<td class="org-left">Types of noise encountered and how to mode them</td>
<td class="org-right">4 -  5</td>
</tr>


<tr>
<td class="org-right">7</td>
<td class="org-left"><a href="file:///Users/danielmcguiness/GitHub/MCI-Source-Files/(BSc - Lecture) Digital Image Processing/Lecture Slides/DIP Slide/codes/Histogram-Operations.corg">Histogram Operations</a></td>
<td class="org-left">Analysis of histogram, both in grey and colour, along with masking and stretching</td>
<td class="org-right">6</td>
</tr>


<tr>
<td class="org-right">8</td>
<td class="org-left"><a href="file:///Users/danielmcguiness/GitHub/MCI-Source-Files/(BSc - Lecture) Digital Image Processing/Lecture Slides/DIP Slide/codes/Morphological-Operations.corg">Morphological Operations</a></td>
<td class="org-left">Morphological operators (i.e., dilation, gradient, &#x2026;)</td>
<td class="org-right">7</td>
</tr>


<tr>
<td class="org-right">9</td>
<td class="org-left"><a href="file:///Users/danielmcguiness/GitHub/MCI-Source-Files/(BSc - Lecture) Digital Image Processing/Lecture Slides/DIP Slide/codes/Blurring-Filters.corg">Blurring Filters</a></td>
<td class="org-left">Types of blurring filters used for noise reduction and smoothing applications</td>
<td class="org-right">8</td>
</tr>


<tr>
<td class="org-right">10</td>
<td class="org-left">Feature Analysis</td>
<td class="org-left">Algorithms used to extract features from images</td>
<td class="org-right">9</td>
</tr>


<tr>
<td class="org-right">11</td>
<td class="org-left">Edge Detection</td>
<td class="org-left">Methods and alhorithms used in detecting edges for computer vision</td>
<td class="org-right">10</td>
</tr>


<tr>
<td class="org-right">12</td>
<td class="org-left"><a href="file:///Users/danielmcguiness/GitHub/MCI-Source-Files/(BSc - Lecture) Digital Image Processing/Lecture Slides/DIP Slide/codes/Neural-Networks-for-Image-Processing.corg">Neural Networks for Image Processing</a></td>
<td class="org-left">A Brief introduction to ANNs for use in image recognition</td>
<td class="org-right">11 - 12</td>
</tr>


<tr>
<td class="org-right">13</td>
<td class="org-left">Group Assignment Presentations</td>
<td class="org-left">Presentations of your group assingments and the following Q &amp; A</td>
<td class="org-right">13 - 15</td>
</tr>


<tr>
<td class="org-right">14</td>
<td class="org-left">Appendix</td>
<td class="org-left">Tangental information related to the main topic</td>
<td class="org-right">&#xa0;</td>
</tr>
</tbody>
</table>


<a id="org1715c47"></a>

## Code Supplement

The Code supplement is a Github webpage dedicated to hosting all the relevant code used in the lecture as
it is not feasible to fit all the content of the code to the slides and it is easier to share this way.

[Visit the Code Supplement Website](https://dtmc0945.github.io/L-MCI-BSc-Digital-Image-Processing/)


<a id="orgddf167e"></a>

## Reading List

The following materials are recommend reading for the coure but by no means are they
mandatory.

**Books**

1.  Young I. "Fundamentals of Image Processing" Delft 1998.
2.  Szeliski R. "Computer Vision: Algorithms and Applications" Springer 2022.
3.  Nixon M. et. al "Feature Extraction and Image Processing for Computer Vision" Academic press 2019
4.  Gonzalez R. "Digital Image Processing" Pearson 2009.

