void settings(){
  size(900, 400);
}

float framePerMs = 60.0f / 1000.0f;

class Cond{
  float speed, origin, xpos;
  int ct, tzone, tcue;
  Cond(int x, int y, int z){
    speed = 100.0f / ((float)y * framePerMs);
    origin = xpos = 400.0f - (speed * 60.0f * ((float)x / 1000.0f));
    ct = z;
    tzone = y;
    tcue = x;
  }

  void move(){
    xpos += speed;
  }

  void setpos(){
    xpos = origin;
  }
};

Cond[] conds = {
  new Cond(0, 80, 1000),
  new Cond(0, 150, 1000),
  new Cond(0, 80, 1500),
  new Cond(0, 150, 1500),
  new Cond(100, 80, 1000),
  new Cond(100, 150, 1000),
  new Cond(100, 80, 1500),
  new Cond(100, 150, 1500)
};

Table table = new Table();
  
