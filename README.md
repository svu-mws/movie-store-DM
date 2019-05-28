Backend API
===========
-   Mining API

    -   Customers who show this movie also show:

        http://localhost:5000/recommended_movies/<movie_name>
        
        example: http://127.0.0.1:5000/recommended_movies/Braveheart        
        
        ```JSON
        {
          "result": ["Indiana Jones and the Raiders of the Lost Ark","Lord of the Rings: The Fellowship of the Ring, The","Monty Python and the Holy Grail","Gladiator","Lethal Weapon","Star Wars","Saving Private Ryan","Godfather, Part II, The","Forrest Gump","Indiana Jones and the Temple of Doom","Godfather, The","Star Wars Episode V: Empire Strikes Back","Shawshank Redemption, The","Matrix, The","Caddyshack","Indiana Jones and the Last Crusade","Die Hard"]
        }
        ```
        
    -   Get personalised recommendations:

        http://localhost:5000/user_movies/<user_id>
        
        example: http://127.0.0.1:5000/user_movies/879061        
        
        ```JSON
        {
          "result": ["Aliens","Alien","Alien Resurrection","Blade Runner","Apollo 13","American Beauty","A Few Good Men","Alien 3","Indiana Jones and the Raiders of the Lost Ark","Austin Powers: The Spy Who Shagged Me"]
        }
        ```
        
    -   Who select the movie:

        http://localhost:5000/movie_selectors/<user_id>
        
        example: http://127.0.0.1:5000/movie_selectors/878908
        
        ```JSON
        {
          "result": "Me"
        }
        ```
        
    -   Who are actor’s friends:

        http://localhost:5000//actor/<actor_name>/friends
        
        example: http://127.0.0.1:5000/actor/Hopkins,%20Anthony/friends
        
        ```JSON
        {
          "result": ["Nicholson, Jack","Jackson, Samuel L.","Ryan, Meg","Jones, Tommy Lee","Washington, Denzel","Spacey, Kevin","Williams, Robin","Willis, Bruce","Hunt, Helen","Murphy, Eddie","Redford, Robert","Jones, James Earl","Pitt, Brad","Pfeiffer, Michelle","Travolta, John","Malkovich, John","Newman, Paul","Hopper, Dennis","Kidman, Nicole","Paltrow, Gwyneth","Judd, Ashley","Martin, Steve","Murray, Bill","Neeson, Liam","Sarandon, Susan","Norton, Edward","Weaver, Sigourney","Myers, Mike","Russo, Rene","Kline, Kevin","Moore, Demi","Robbins, Tim","Hunter, Holly","O'Toole, Peter","Smith, Will","Walken, Christopher","Witherspoon, Reese","Quaid, Dennis"]
        }
        ```
