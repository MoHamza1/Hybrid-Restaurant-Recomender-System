# im sorry this isnt documented better, it was put together quite hasely
import pandas as pd
import json
from surprise import KNNBaseline, Reader, Dataset
from surprise.model_selection import cross_validate

res_id_name_dict = {}
with open("reduced_dataset/users.json", "r") as infile:
    _temp = infile.read()
    _temp = json.loads(_temp)
    user_white_list = _temp["a"]
infile.close()


def get_n_recoms(algo, bus_white_list, target_user, n):
    recomendations = []
    for business in bus_white_list:
        temp = algo.predict(target_user, business, verbose=False)
        recomendations.append({"business": business, "estimate": temp.est})

    recomendations.sort(key=lambda x: x["estimate"], reverse=True)

    recomendations = recomendations[:min(len(recomendations), n)]

    for recom in recomendations:
        print(f'\n{res_id_name_dict[recom["business"]]},\n\t{int(recom["estimate"] * 20)}% Match for you')


# all keywords that apply to businesses in the domain.
all_cats = ['Desserts', 'Ice Cream & Frozen Yogurt', 'Pretzels', 'Bakeries', 'Fast Food', 'Restaurants',
            'Specialty Food',
            'Food Stands', 'Mexican', 'American (New)', 'Burgers', 'American (Traditional)', 'Breakfast & Brunch',
            'Salad',
            'Lebanese', 'Middle Eastern', 'Halal', 'Imported Food', 'Ethnic Food', 'Coffee & Tea', 'Cafes',
            'Sandwiches',
            'Coffee Roasteries', 'Nightlife', 'Coffeeshops', 'Bagels', 'Donuts', 'Food Court', 'Juice Bars & Smoothies',
            'Hot Dogs', 'Chicken Shop', 'Chinese', 'Soup', 'Vietnamese', 'Thai', 'Tex-Mex', 'Tacos', 'Chicken Wings',
            'Seafood', 'Tea Rooms', 'Pubs', 'Bars', 'Beer', 'Gastropubs', 'British', 'Fish & Chips', 'Dive Bars',
            'Pizza',
            'Italian', 'Gluten-Free', 'Bubble Tea', 'Delis', 'Canadian (New)', 'Cajun/Creole', 'Food Delivery Services',
            'Street Vendors', 'Steakhouses', 'Beer Bar', 'Cocktail Bars', 'Wine Tasting Room', 'Sushi Bars', 'Barbeque',
            'Argentine', 'Poutineries', 'Comfort Food', 'Custom Cakes', 'Patisserie/Cake Shop', 'Do-It-Yourself Food',
            'Japanese', 'Dim Sum', 'Vegetarian', 'Ramen', 'Vegan', 'Diners', 'Filipino', 'Wraps', 'Hawaiian',
            'Caribbean',
            'Malaysian', 'Asian Fusion', 'Shaved Ice', 'Themed Cafes', 'Shaved Snow', 'Spanish', 'French', 'Szechuan',
            'Pasta Shops', 'Buffets', 'Tapas Bars', 'Wine Bars', 'Pan Asian', 'Indian', 'Portuguese', 'Fondue',
            'Korean',
            'Greek', 'Gelato', 'Poke', 'Burmese', 'Soul Food', 'Swiss Food', 'Modern European', 'Irish Pub',
            'Food Trucks',
            'Noodles', 'Cheesesteaks', 'Turkish', 'Teppanyaki', 'Cupcakes', 'Taiwanese', 'Sri Lankan', 'Pakistani',
            'Cheese Shops', 'Singaporean', 'Afghan', 'Tapas/Small Plates', 'Cantonese', 'Waffles', 'Pita', 'Brewpubs',
            'Scandinavian', 'Macarons', 'Puerto Rican', 'German', 'African', 'Hot Pot', 'Persian/Iranian', 'Cuban',
            'Falafel',
            'Brazilian', 'Beer Gardens', 'Colombian', 'Whiskey Bars', 'Hungarian', 'Moroccan', 'Dumplings', 'Russian',
            'Kebab',
            'Japanese Curry', 'Eritrean', 'Shanghainese', 'Honey', 'Pancakes', 'Himalayan/Nepalese', 'Ukrainian',
            'Popcorn Shops', 'Egyptian', 'Empanadas', 'Polish', 'Hookah Bars', 'Salvadoran', 'Tiki Bars', 'Indonesian',
            'Kombucha', 'Tui Na', 'Food Tours', 'Pub Food', 'Scottish', 'Udon', 'Beer Tours', 'Honduran', 'Polynesian',
            'Uzbek', 'Beer Garden', 'Haitian', 'Syrian']


def content_BF(_desired_category, userID):
    bus_white_list = []
    user_liked_previous = []
    user_keyword_likes = _desired_category

    # open reviews corpus and find business user liked before.
    if len(_desired_category) == 0:
        with open("reduced_dataset/reviews.csv", "r") as infile:
            while True:
                temp = infile.readline()
                if not temp:
                    break
                else:
                    temp = temp.split(",")
                    if temp[0] == userID and int(temp[2][:1]) > 3:
                        user_liked_previous.append(temp[1])

        infile.close()

        # open restaurant corpus and find their keywords
        with open("reduced_dataset/restaurants.json", "r") as infile1:
            while True:
                _temp = infile1.readline()

                if _temp:
                    _temp = json.loads(_temp)
                    if _temp["alias"][0] in user_liked_previous:
                        for x in _temp["categories"]:
                            if x not in user_keyword_likes and x in all_cats:
                                user_keyword_likes.append(x)
                else:
                    break
        infile1.close()
    # open again and identify all the restaurants matchnig previous user sea
    with open("reduced_dataset/restaurants.json", "r") as infile1:
        while True:
            _temp = infile1.readline()

            if _temp:
                _temp = json.loads(_temp)
                if _temp["review_count"] > 100:
                    # finding the eucledian distance between the two sets instead of computing TF-IDF since keywords are not that many.
                    if len(set.intersection(set(user_keyword_likes), set(_temp["categories"]))) >= 1:
                        bus_white_list.append(_temp["alias"][0])
                        res_id_name_dict[_temp["alias"][0]] = _temp["name"]
            else:
                break
    infile1.close()
    # extracted list of businesses that match both search criteria and previous user behaviour.
    # these are the businesses that we already like
    # and others with similar keywords.
    # now lets use this to get some recomendations from others who reviewed the venues.
    return bus_white_list


def Collaberative_filtering(bus_white_list, mode):
    #  Split Train and fit
    # Load the reviews up
    df = pd.read_csv("reduced_dataset/reviews.csv")
    # Remove reviews that dont involve businesses whos key-words are not desired.
    df = df[df['business_id'].isin(bus_white_list)]
    df = df[df['user_id'].isin(user_white_list)]
    df = df.sort_values(by=['business_id'])
    # Will be removed when the above todos have been implemented.

    reader = Reader(rating_scale=(0.0, 5.0))
    data = Dataset.load_from_df(df[['user_id', 'business_id', 'stars']], reader)
    trainset = data.build_full_trainset()
    algo = KNNBaseline(k=40, min_k=2, sim_options={'user_based': False}, verbose=False)

    if mode:
        x = cross_validate(algo, data, measures=['MAE'])
        return x["test_mae"]
    else:
        algo.fit(trainset)
        return trainset, algo


def userInterface():
    while True:
        logged_out = True
        while logged_out:
            print("Please enter the userID you have, or enter 'O' to get one that exists.")
            resp = input(">")
            if resp == "O":
                print("Here you go:\nwV4uhvJGok8nOR56Ex5mAg\n")
            elif len(resp) == 22:
                # Check if this user exists.
                target_user = resp
                if target_user in user_white_list:
                    print("Logged in successfully!")
                    print("Please wait a few moments, we will be ready soon...")
                    bus_white_list = content_BF([], target_user)
                    trainset, algo = Collaberative_filtering(bus_white_list, False)
                    print("Done! ready to use::")
                    logged_out = not logged_out
                else:
                    print("User doesn't exist")

        while not logged_out:
            print("\nEnter :ESC now to log out of this userID.")
            print(
                "Would you like to:\n\t1.Get top 10 recomendations\n\t2.Get custom number of top recomendations\n\t3.Configure search parameters\n")
            resp = input(">")
            if resp == "1":
                get_n_recoms(algo, bus_white_list, target_user, 10)


            elif resp == "2":
                while True:
                    print("How many recomendations do you want?")
                    x = input(">>")
                    try:
                        x = int(x)
                        if x > 0:
                            get_n_recoms(algo, bus_white_list, target_user, x)
                            break
                    except ValueError:
                        print("How many recomendations do you want? enter an integer!")

            elif resp == "3":

                print("Here are all the categories:")
                for loop in range(0, len(all_cats), 2):
                    print(f"{loop + 1}.{all_cats[loop]}\t{loop + 2}.{all_cats[loop + 1]}")
                print("Enter the number of the parameter you want; enter 'S' to search when you are done.")
                desired_cat = []
                while True:
                    x = input(">>")
                    if x == "S":
                        if len(desired_cat) == 0:
                            desired_cat = []
                        break
                    else:
                        try:
                            desired_cat.append(all_cats[int(x) - 1])
                        except ValueError:
                            print("Enter the number of the parameter you want; enter 'S' to search when you are done.")

                print("You've selected:", desired_cat)
                print("Updating the system. please wait a moment...")
                bus_white_list = content_BF(desired_cat, target_user)
                trainset, algo = Collaberative_filtering(bus_white_list, False)
                print("Done! ready to use::")
            elif resp == ":ESC":
                logged_out = not logged_out


userInterface()
