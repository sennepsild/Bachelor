using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Game  {

    Player player1;
    Player player2;
    Ball ball;
    int sizeX;
    int sizeY;

    public int rightPlayerLife = 3;
    public int leftPlayerLife = 3;



    public Game(int sizeX,int sizeY)
    {
        this.sizeX = sizeX;
        this.sizeY = sizeY;
        ball = new Ball( sizeX, sizeY, this);
    }

    public void Update()
    {
        ball.move();
    }

    public Vector2 getBallPos()
    {
        return ball.position;

    }



}
