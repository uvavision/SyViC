using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Traverse_Nodes : MonoBehaviour
{
    GameObject objToSpawn;
    Vector3 zeroPosition = new Vector3(0,0,0);
    Quaternion zeroRot = Quaternion.Euler(new Vector3(-10,0,0));

    // Start is called before the first frame update
    void Start()
    {


        this.gameObject.AddComponent<Rigidbody>();
        Transform[] transforms = this.GetComponentsInChildren<Transform>();
 
        foreach(Transform t in transforms)
        {


            if ((t.gameObject.name == "left_foot") || (t.gameObject.name == "right_foot"))
            {
                // Debug.Log ("Found " + t);

                // spawns object
                objToSpawn = new GameObject(t.gameObject.name + "_extra");
                objToSpawn.transform.SetPositionAndRotation(new Vector3(t.transform.position.x+0.1f, t.transform.position.y+0.1f, t.transform.position.z+0.1f), Quaternion.Euler(new Vector3(t.rotation.eulerAngles.x, t.rotation.eulerAngles.y - 5, t.rotation.eulerAngles.z)));

                // add Components
                MeshFilter meshFilter = objToSpawn.AddComponent<MeshFilter>();
                BoxCollider boxCollider = objToSpawn.AddComponent<BoxCollider>();
                boxCollider.center = new Vector3(0.0f, 0.0f, 10.0f); // Vector3(0.0f, -2.4f, 9.5f);
                boxCollider.size = new Vector3(50, 10, 50);
                MeshRenderer meshRenderer = objToSpawn.AddComponent<MeshRenderer>();

               // sets the obj's parent to the obj that the script is applied to
                objToSpawn.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
                objToSpawn.transform.SetParent(t);
            }

            if ((t.gameObject.name == "left_middle1") || (t.gameObject.name == "right_middle1"))
            {

                // spawns object
                objToSpawn = new GameObject(t.gameObject.name + "_extra");
                // objToSpawn.transform.position = t.position;
                Debug.Log (objToSpawn.transform.position);
                Debug.Log (t.transform.position);

                objToSpawn.transform.SetPositionAndRotation(new Vector3(t.transform.position.x+0.1f, t.transform.position.y+0.1f, t.transform.position.z+0.1f), Quaternion.Euler(new Vector3(t.rotation.eulerAngles.x, t.rotation.eulerAngles.y - 10, t.rotation.eulerAngles.z)));
                Debug.Log (objToSpawn.transform.position);
                Debug.Log (t.transform.position);


                // add Components
                MeshFilter meshFilter = objToSpawn.AddComponent<MeshFilter>();
                BoxCollider boxCollider = objToSpawn.AddComponent<BoxCollider>();
                boxCollider.center = new Vector3(0.0f, -3.0f, 9.5f);
                boxCollider.size = new Vector3(5, 5, 10);
                MeshRenderer meshRenderer = objToSpawn.AddComponent<MeshRenderer>();

               // sets the obj's parent to the obj that the script is applied to
                objToSpawn.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
                objToSpawn.transform.SetParent(t);
            }
        }
        
    }

    // Update is called once per frame
    void Update()
    {
        // objToSpawn.transform.SetPositionAndRotation(zeroPosition, zeroRot);

        Transform[] transforms = this.GetComponentsInChildren<Transform>();
 
        foreach(Transform t in transforms)
        {
            // Debug.Log (t.gameObject.name);
            if ((t.gameObject.name == "left_foot_extra") || (t.gameObject.name == "right_foot_extra"))
            {
                Debug.Log ("Found " + t.transform.parent.transform.name + " - " + t.transform.parent.transform.position);

                t.transform.SetPositionAndRotation(t.transform.parent.transform.position, Quaternion.Euler(new Vector3(t.transform.parent.rotation.eulerAngles.x, t.transform.parent.rotation.eulerAngles.y, t.transform.parent.rotation.eulerAngles.z)));

            }
        }
    }
}
