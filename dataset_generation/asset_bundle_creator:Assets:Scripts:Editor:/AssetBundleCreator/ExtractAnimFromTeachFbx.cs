using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using SubalternGames;
using Object = UnityEngine.Object;
using Debug = UnityEngine.Debug;
using Logger = Logging.Logger;

using UnityEditor.Presets;

public class ExtractAnimFromTeachFbx
{
    public ExtractAnimFromTeachFbx()
    {
        Debug.Log("just called ExtractAnimFromTeachFbx");
        DirectoryInfo d = new DirectoryInfo("Assets/Resources"); //Assuming Test is your Folder
        FileInfo[] Files = d.GetFiles("*.fbx"); //Getting Text files

        Debug.Log("List of files:");
        Debug.Log(Files);

        foreach(FileInfo file in Files )
        {
            string str =file.Name;

            //Extract the animations from the fbxs and place the in resources to be
            //used for overrideAnimation        
            AnimationClip orgClip = (AnimationClip)AssetDatabase.LoadAssetAtPath("Assets/Resources/" + file.Name,typeof(AnimationClip) );
            SerializedObject serializedClip = new SerializedObject(orgClip);

            //Save the clip
            AnimationClip placeClip = new AnimationClip();
            if( !Resources.Load("Assets/Resources/" + file.Name.Replace(".fbx", ".anim")))
            {
                var newAssetName = "Assets/Resources/" + file.Name.Replace(".fbx", ".anim");
                EditorUtility.CopySerialized(orgClip,placeClip);
                AssetDatabase.CreateAsset(placeClip, newAssetName);
                AssetImporter.GetAtPath(newAssetName).SetAssetBundleNameAndVariant(file.Name.Replace(".fbx", "_anim"), "");
                AssetDatabase.Refresh();

                Debug.Log (newAssetName);
            }
        }
    }

    public static void SourceFileToAssetBundles()
    {
        Debug.Log("just called ExtractAnimFromTeachFbx");
        DirectoryInfo d = new DirectoryInfo("Assets/Resources"); //Assuming Test is your Folder
        FileInfo[] Files = d.GetFiles("*.fbx"); //Getting Text files

        Debug.Log("List of files:");
        Debug.Log(Files);

        foreach(FileInfo file in Files )
        {
            string str =file.Name;

            //Extract the animations from the fbxs and place the in resources to be
            //used for overrideAnimation        
            AnimationClip orgClip = (AnimationClip)AssetDatabase.LoadAssetAtPath("Assets/Resources/" + file.Name,typeof(AnimationClip) );
            SerializedObject serializedClip = new SerializedObject(orgClip);

            //Save the clip
            AnimationClip placeClip = new AnimationClip();
            if( !Resources.Load("Assets/Resources/" + file.Name.Replace(".fbx", ".anim")))
            {
                var newAssetName = "Assets/Resources/" + file.Name.Replace(".fbx", ".anim");
                EditorUtility.CopySerialized(orgClip,placeClip);
                AssetDatabase.CreateAsset(placeClip, newAssetName);
                AssetImporter.GetAtPath(newAssetName).SetAssetBundleNameAndVariant(file.Name.Replace(".fbx", "_anim"), "");
                AssetDatabase.Refresh();

                Debug.Log (newAssetName);
            }
        }
    }

}