using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Pong : MonoBehaviour {
    Game game;
    public GameObject ball;
    float fps = 1;
    float time = 0;

	// Use this for initialization
	void Start () {
        game = new Game(2,1);
	}
	
	// Update is called once per frame
	void Update () {
        time += Time.deltaTime;

        if (time >= 1 / fps)
        {
            
            game.Update();
            time = 0;
        }
        ball.transform.Translate(new Vector3( game.getBallPos().x, game.getBallPos().y, 0));

        print(game.getBallPos());
		
	}
}
