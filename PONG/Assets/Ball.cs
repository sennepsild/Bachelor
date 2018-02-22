using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ball  {
    public Vector2 position;
    Vector2 direction;
    float speed = 1;
    float height;
    float width;

    Game game;

    public Ball( int width, int height, Game game)
    {
        this.game = game;
        this.height = height / 2;
        this.width = width/2;
        direction = new Vector2(-1, -0.5f);
        position = new Vector2();
    }

    public void move()
    {
        position = position + direction * speed;
        if (position.y >= height||position.y <= -height)
            direction.y = direction.y * -1;

        if (position.x >= width)
            Ballout("right");
        if (position.x <= -width)
            Ballout("left");

    }

    void Ballout(string side)
    {
        if (side == "left")
            game.leftPlayerLife--;
        else
            game.rightPlayerLife--;

        position = new Vector2();
    }
	
}
