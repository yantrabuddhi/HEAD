//OpenCV libraries
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//ROS
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <pi_face_tracker/Face.h>
#include <pi_face_tracker/Faces.h>
#include <face_id/f_id.h>
#include <face_id/faces_ids.h>

#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
//openbr
#include <openbr/openbr_plugin.h>
#include <QMutex>
#include <QString>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <mutex>
using namespace std;
//using namespace cv;

//ASSUMES SAME IMAGE RESOLUTION IMAGE REFERENCE AS CMT TRACKER

//load gallery txt file with file,person_name strings to create a mapping
//subscribe to camera and tracked faces
//if recognized faces in camera image overlap with tracked faces then map them in ros msg

QSharedPointer<br::Transform> transformb;// = br::Transform::fromAlgorithm("FaceRecognition");
QSharedPointer<br::Distance> distanceb;// = br::Distance::fromAlgorithm("FaceRecognition");
std::mutex z_mutex;
std::string gDir,gInfo;

ros::Publisher faces_pub;
br::TemplateList target;

string face_cascade_name = "haarcascade_frontalface_alt.xml";
string gallery_path;
std::map<string,string> file2name;
//String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
cv::CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
std::vector<cv::Rect> faces;
struct face_pix{
  int id;
  int fx;
  int fy;
};
std::vector<face_pix>frv;

std::vector<cv::Mat> getFaces(cv::Mat frame_gray)
{
  //
  std::vector<cv::Mat> fcs;
  cv::equalizeHist( frame_gray, frame_gray );
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(64, 64) );//30,30
  for( size_t i = 0; i < faces.size(); i++ )
  {
    //
    /*
    faces[i].x=faces[i].x-10;
    faces[i].y=faces[i].y-10;
    faces[i].width=faces[i].width+20;
    faces[i].height=faces[i].height+20;
    */
    cv::Rect rct(faces[i].x-faces[i].width*0.25,faces[i].y-faces[i].height*0.25,
      faces[i].width+faces[i].width*2*0.25,faces[i].height+faces[i].height*2*0.25);
    if (rct.x<0)rct.x=0;
    if (rct.y<0)rct.y=0;
    if (rct.x+rct.width>639)rct.width=639-rct.x;
    if (rct.y+rct.height>479)rct.height=479-rct.y;
    cv::Mat faceROI = frame_gray( rct );//faces[i]
    //cv::resize(faceROI,faceROI,cv:Size(80,))
    fcs.push_back(faceROI);
  }
  return fcs;
}

bool is_point_in_rect(float x,float y,cv::Rect rct)
{
  return ((x>=rct.x) && (x<=rct.x+rct.width) && (y>=rct.y) && (y<=rct.y+rct.height));
}
bool rects_overlap(cv::Rect r1,cv::Rect r2)
{
  return (
    is_point_in_rect(r1.x,r1.y,r2)||
    is_point_in_rect(r1.x+r1.width,r1.y,r2)||
    is_point_in_rect(r1.x,r1.y+r1.height,r2)||
    is_point_in_rect(r1.x+r1.width,r1.y+r1.height,r2)
  );
}

int get_overlap_id(cv::Rect rct)
{
  std::lock_guard<std::mutex> guard(z_mutex);
  for (int  i=0;i<frv.size();i++)
  {
    if (is_point_in_rect(frv[i].fx,frv[i].fy,rct))
    {
      return frv[i].id;
    }
  }
  return 0;
}

std::string get_name_from_gallery(int idx)
{
  //check if file.name is same as name in gallery
  std::map<string,string>::iterator it;
  it=file2name.find(target[idx].file.name.toStdString());
  if (it != file2name.end())
    return file2name[target[idx].file.name.toStdString()];
  return "stranger";
}

void image_cb(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
  cv::Mat img = cv_ptr->image;
  cv::Mat img_gray;
  //ROS_INFO("I heard: [%s]", msg->data.c_str());
  //find faces and pass each face with co-ordinates for recognition
  cv::cvtColor(img, img_gray, CV_BGR2GRAY);
  std::vector<cv::Mat> fcs=getFaces(img_gray);
  face_id::faces_ids fc_ids;
  //std::vector<face_id::face_id> fc_ids;
/*
  if (fcs.size()<1){cout<<"no image...\n";return;}
  cout<<"image..."<<fcs.size()<<"\n";
  cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
  cv::imshow( "Display window", fcs[0] );                   // Show our image inside it.
  cv::waitKey(1);
  return;
*/
  for( size_t i = 0; i < fcs.size(); i++ )
  {
    cv::Mat mt=fcs[i].clone();//mat needs to be continuous
    br::Template query(mt);
    query >> *transformb;//why error here?
    // Compare templates
    QList<float> scores = distanceb->compare(target, query);
    //FORM ROS MSG
    float score=scores[0];
    int index=0;
    for (int a=1;a<scores.size();a++)
    {
      if (scores[a]>score)
      {
        score=scores[a];
        index=a;
      }
    }
    //update message to send
    face_id::f_id fid;
    fid.id=get_overlap_id(faces[index]);
    fid.name=((score>0.8)?get_name_from_gallery(index):"stranger");
    fid.confidence=score;
    if (score>0.8)fc_ids.faces.push_back(fid);
    //std::cout<<"\n"<<target[index].file.name.toStdString()<<"\n";
  }
  //publish message
  if (fc_ids.faces.size()<1) return;
  faces_pub.publish(fc_ids);
  /*
  face_id::faces_ids fi;
  fi.faces=fc_ids; wrong
  faces_pub.publish(fi);
  */
}

string trim(const string& str)
{
    size_t first = str.find_first_not_of(' ');
    if (string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

void map_image2name(string line)
{
  //split line in 2
  string file,name;
  unsigned int pos = line.find( "," );
  if (pos<1) return;
  file= trim(line.substr( 0, pos ));
  name= trim(line.substr(pos+1,line.length()-pos-1));
  file2name[file]=name;
}

void faces_cb(const pi_face_tracker::FacesConstPtr& msg)
{
  const float pic_width=640.0;//pixels
  const float pic_height=480.0;
  const float fov=1.42;//radians
  //const float fsz=0.17;//meters face width
  const double k_const = (double)pic_width / (double) (2.0*(tan(fov/2.0)));
  float xx,yy,zz;
  face_pix fr;
  //need tracking id and rect
  std::lock_guard<std::mutex> guard(z_mutex);
  frv.clear();
  for (int i=0;i<msg->faces.size();i++)
  {
    //
    fr.id=msg->faces[i].id;
    //convert x,y to pixels from 3d xyz
    xx=msg->faces[i].point.x;//z
    yy=msg->faces[i].point.y;//x
    zz=msg->faces[i].point.z;//y
    double dp=xx/k_const;
    fr.fy=(pic_width/2.0)-(zz/dp);
    fr.fx=(pic_height/2.0)-(yy/dp);
    frv.push_back(fr);
  }
}

void load_name_image_map(string gfile)
{
  //open gallery file from path
  string line;
  ifstream gallery_file(gfile);
  //gallery_file.open();
  while ( getline (gallery_file,line) )
  {
    map_image2name(line);
  }
  gallery_file.close();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "faceid");
  ros::NodeHandle n;
  //param to know path of gallery and gallery description file
  n.param<string>("gallery_dir",gDir,"gallery_dir");
  n.param<string>("gallery_info_file",gInfo,"gallery_dir/gallery_info.txt");
  n.param<string>("haarcascade_frontalface_alt",face_cascade_name,"haarcascade_frontalface_alt.xml");
  if( !face_cascade.load( face_cascade_name ) ){ cout<<"--(!)Error loading face cascade\n"; return -1; };
  //if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  //Load image name map
  load_name_image_map(gInfo);
  if (file2name.size()<1){cout<<"--(!)Error loading Gallery Info file.\n"; return -1;};
  //
  br::Context::initialize(argc, argv);

  // Retrieve classes for enrolling and comparing templates using the FaceRecognition algorithm
  transformb = br::Transform::fromAlgorithm("FaceRecognition");
  distanceb = br::Distance::fromAlgorithm("FaceRecognition");

  // Initialize templates
  target = br::TemplateList::fromGallery(gDir.c_str());
  // Enroll templates
  br::Globals->enrollAll = true; // Enroll 0 or more faces per image
  target >> *transformb;
  br::Globals->enrollAll = false;
  //
  ros::Subscriber sub = n.subscribe("/camera/image_raw", 2, image_cb);
  ros::Subscriber sub_face = n.subscribe("/camera/face_locations", 1, &faces_cb);
  //need rect of face
  faces_pub = n.advertise<face_id::faces_ids>("/camera/face_recognition", 1);
  ros::spin();
  br::Context::finalize();
  return 0;
}
